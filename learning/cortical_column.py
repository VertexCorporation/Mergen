"""
╔══════════════════════════════════════════════════════════════════════╗
║        MERGEN — CorticalColumn (Çok Katmanlı Spiking Substrat)       ║
║                                                                      ║
║  Biyolojik referans: İnsan neokorteksi 6 katmanlı kortikal          ║
║  kolonlardan oluşur. Her kolon, girdiden çıktıya kadar hiyerarşik   ║
║  soyutlama yapar. Mergen için 4 temel katmana sadeleştirildi.        ║
║                                                                      ║
║  Katman mimarisi:                                                    ║
║    L4  (Granüler)        — Thalamik girdi alır, özellik çıkarımı    ║
║    L23 (Supragranüler)   — Lateral entegrasyon, içsel işleme         ║
║    L5  (İnfragranüler)   — Çıkış: neural_intent üretir              ║
║    L6  (Multiformis)     — [Pasif] Gelecek: geri besleme tahmini    ║
║                                                                      ║
║  API Uyumluluğu: HybridHebbianLearner ile tamamen aynı arayüz.      ║
║  brain.py ve limbic_executive_layer.py'de tek satır değişir.        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

from learning.gradients import SpikingActivation
from learning.stdp import STDPMechanism
from learning.rl_agent import DopamineModulator

# Özel token maskesi (HybridHebbianLearner ile aynı sabit)
NUM_MASKED_TOKENS = 3


# ═══════════════════════════════════════════════════════════════════
#  CorticalLayer — Tek Kortikal Katman
#  (HybridHebbianLearner mantığının ayrıştırılmış, tek katman hali)
# ═══════════════════════════════════════════════════════════════════

class CorticalLayer(nn.Module):
    """
    Tek bir kortikal işleme katmanı.

    Her katman şunları içerir:
    • Sinaptik ağırlıklar   (nn.Parameter, STDP ile öğrenilir)
    • Pre/Post izleri       (STDP zamanlama belleği)
    • Eligibility trace     (dopamin bekleme sinyali)
    • Firing rate EMA       (homeostaz için)
    • k-WTA inhibisyonu     (seyrek kodlama)
    • STDP + Dopamin öğrenmesi
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        dt: float = 1.0,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        tau_eligibility: float = 25.0,
        A_ltp: float = 0.005,
        A_ltd: float = 0.003,
        w_max: float = 1.0,
        w_min: float = 0.0,
        target_input_sum: float = 40.0,
        scaling_speed: float = 0.001,
        target_firing_rate: float = 0.1,
        dopamine_threshold: float = 0.01,
        spike_threshold: float = 1.0,
        lateral_k: int = 10,
        ema_decay: float = 0.99,
        mask_first_n: int = 0,
        device: str = 'cpu',
        # Topolojik Yanal Bağlantı (Mexican Hat) — sadece L23 için
        # None → devre dışı (L4, L5 bu özelliği kullanmaz)
        # (H, W) → n_out == H*W olmalı; Mexican Hat W_lat matrisi oluşturulur
        topology_grid: Optional[Tuple[int, int]] = None,
        # Mexican Hat parametreleri (biyolojik değerlere göre ayarlandı)
        lat_sigma_exc: float = 2.0,   # Uyarıcı komşuluk yarıçapı (grid adımı)
        lat_sigma_inh: float = 5.0,   # İnhibitör bölge yarıçapı
        lat_A_exc: float = 0.5,       # Uyarıcı genlik
        lat_A_inh: float = 0.25,      # İnhibitör genlik
        lat_spectral_target: float = 0.9,  # Spektral normalizasyon hedefi
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.dt = dt
        self.A_ltp = A_ltp
        self.A_ltd = A_ltd
        self.w_max = w_max
        self.w_min = w_min
        self.target_input_sum = target_input_sum
        self.scaling_speed = scaling_speed
        self.target_firing_rate = target_firing_rate
        self.dopamine_threshold = dopamine_threshold
        self.lateral_k = lateral_k
        self.ema_decay = ema_decay
        self.mask_first_n = mask_first_n  # Özel token maskesi (sadece L5 çıkış katmanında)

        # Precomputed decay factors
        self._decay_pre = 1.0 - (dt / tau_pre)
        self._decay_post = 1.0 - (dt / tau_post)
        self._decay_elig = 1.0 - (dt / tau_eligibility)

        # Sinaptik ağırlıklar
        self.weights = nn.Parameter(
            torch.rand(n_in, n_out, device=device) * 0.3
        )

        # Dinamik durum buffer'ları
        self.register_buffer('trace_pre',        torch.zeros(n_in,       device=device))
        self.register_buffer('trace_post',       torch.zeros(n_out,      device=device))
        self.register_buffer('eligibility',      torch.zeros(n_in, n_out, device=device))
        self.register_buffer('firing_rate_ema',  torch.zeros(n_out,      device=device))

        # Bileşenler
        self.spike_fn = SpikingActivation(threshold=spike_threshold)

        # Her katmanın kendi STDP ve dopamin nesnesi var
        self.stdp = STDPMechanism(learning_rate=A_ltp, tau_trace=tau_pre, dt=dt)
        self.dopamine = DopamineModulator(gamma=0.99, lr_critic=0.1)

        # ── Topolojik Yanal Bağlantı (Mexican Hat) ──────────────────────────
        # topology_grid=(H, W) verilmişse n_out == H*W olmalı.
        # W_lat: (n_out, n_out) sabit tampon. Öğrenilmez; fiziksel mesafeye dayalı.
        self._has_topology = False
        if topology_grid is not None:
            H, W = topology_grid
            if H * W != n_out:
                raise ValueError(
                    f"topology_grid={topology_grid} → H*W={H*W} != n_out={n_out}. "
                    f"n_out, H×W'ye eşit olmalıdır."
                )
            self._has_topology = True
            self._grid_H = H
            self._grid_W = W
            W_lat = self._build_mexican_hat(
                H, W,
                sigma_exc=lat_sigma_exc,
                sigma_inh=lat_sigma_inh,
                A_exc=lat_A_exc,
                A_inh=lat_A_inh,
                spectral_target=lat_spectral_target,
                device=device,
            )
            # register_buffer: .to(device) ve .state_dict() ile yönetilir
            self.register_buffer('W_lat', W_lat)

        # Telemetri
        self._step_count = 0
        self._da_event_count = 0
        self._last_rpe = 0.0
        self._last_ltp_mag = 0.0
        self._last_ltd_mag = 0.0
        self._last_delta_w = 0.0
        self._last_sparsity = 0.0

    @staticmethod
    def _build_mexican_hat(
        H: int,
        W: int,
        sigma_exc: float,
        sigma_inh: float,
        A_exc: float,
        A_inh: float,
        spectral_target: float,
        device: str,
    ) -> torch.Tensor:
        """
        Mexican Hat (Difference of Gaussians) yanal ağırlık matrisi oluştur.

        Biology: L23 kortikal nöronlar, yakın komşularını uyarır (excitatory
        collaterals), uzak komşuları ise GABAerjik internöronlar aracılığıyla
        inhibe eder. Sonuç: lokal excitation + global inhibition → seyrek,
        topolojik temsil.

        Args:
            H, W:      Grid boyutları. n_out = H * W.
            sigma_exc: Uyarıcı Gauss genişliği (grid adımı biriminde).
            sigma_inh: İnhibitör Gauss genişliği.
            A_exc:     Uyarıcı genlik.
            A_inh:     İnhibitör genlik.
            spectral_target: Spektral normalizasyon hedefi (ör. 0.9).
                       Bunu aşan matrisler ölçeklenir → runaway excitation önlenir.
            device:    torch device string.

        Returns:
            W_lat: (H*W, H*W) float tensör, diyagonal sıfır (öz-geri besleme yok).
        """
        N = H * W
        # 2D koordinatlar
        coords = torch.zeros(N, 2)
        for idx in range(N):
            coords[idx, 0] = idx // W   # satır
            coords[idx, 1] = idx % W    # sütun

        # Karesel Öklidyen mesafe matrisi (N x N)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 2)
        dist_sq = (diff ** 2).sum(dim=-1)                  # (N, N)

        # Mexican Hat = excitatory Gaussian - inhibitory Gaussian
        exc = A_exc * torch.exp(-dist_sq / (2.0 * sigma_exc ** 2))
        inh = A_inh * torch.exp(-dist_sq / (2.0 * sigma_inh ** 2))
        W_lat = exc - inh                                  # (N, N)

        # Öz-bağlantı (self-excitation) sıfırla
        W_lat.fill_diagonal_(0.0)

        # Spektral normalizasyon: maksimum özdeğer > spectral_target ise ölçekle
        # Not: power iteration (tam SVD değil) CPU'da bile hızlı çalışır.
        try:
            # En büyük tekil değer (approx.) — 20 iterasyon yeterli
            v = torch.randn(N, device='cpu')
            for _ in range(20):
                v = W_lat @ v
                norm = v.norm()
                if norm < 1e-10:
                    break
                v = v / norm
            spectral_radius = (W_lat @ v).norm().item()
            if spectral_radius > spectral_target:
                W_lat = W_lat * (spectral_target / spectral_radius)
        except Exception as spec_err:
            # Normalizasyon başarısız olursa basit L∞ normalizasyonu
            w_max = W_lat.abs().max().item()
            if w_max > 0:
                W_lat = W_lat * (spectral_target / w_max)

        return W_lat.to(device)

    def forward(self, pre_spikes: torch.Tensor, spiking: bool = True) -> torch.Tensor:
        """
        İleri besleme: Matmul → (Mexican Hat Lateral) → Spike → k-WTA → çıkış.

        Topolojik katmanlarda (topology_grid != None) iki aşamalı spike:
          1. İlk spike (membran potansiyelinden)
          2. Yanal etkileşim: sadece aktif (ateşlenmiş) nöronlar komşularını etkiler
          3. Rafine membran potansiyeli = ilk membran + lateral giriş
          4. Son spike (rafine potansiyelden)
          5. k-WTA global seyreklik uygulaması

        Bu tasarım, Mexican Hat'in amacına (sadece aktif nöronların komşularını
        etkilemesi) sadık kalır ve runaway excitation'u önler.
        """
        sq = pre_spikes.dim() == 1
        if sq:
            pre_spikes = pre_spikes.unsqueeze(0)

        membrane = torch.matmul(pre_spikes, self.weights)

        # Özel token maskesi (sadece çıkış katmanında etkin)
        if self.mask_first_n > 0 and membrane.shape[-1] > self.mask_first_n:
            membrane[..., :self.mask_first_n] = -1e9

        # ── Topolojik Yanal Etkileşim (Mexican Hat) ────────────────────────
        # Sadece topology_grid verilmiş katmanlarda aktif (L23)
        if self._has_topology:
            # Aşama 1: İlk spike — membran potansiyelinden
            initial_spikes = self.spike_fn(membrane)

            # Aşama 2: Yanal giriş — SADECE ateşlenen nöronlar komşularını etkiler
            # initial_spikes: (batch, n_out) — ateşlemeyenler zaten 0
            lateral_input = torch.matmul(initial_spikes, self.W_lat)  # (batch, n_out)

            # Aşama 3: Rafine membran = ham membran + yanal katkı
            membrane = membrane + lateral_input

            # Token maskesini yeniden uygula (lateral input maskeyi bozmuş olabilir)
            if self.mask_first_n > 0 and membrane.shape[-1] > self.mask_first_n:
                membrane[..., :self.mask_first_n] = -1e9

        # ── Son Aktivasyon ─────────────────────────────────────────────────
        if spiking:
            out = self.spike_fn(membrane)
        else:
            # Graded potential (ReLU applied to keep values non-negative)
            out = torch.relu(membrane)

        # k-Winners-Take-All global seyreklik (topoloji sonrası uygulanır)
        if self.lateral_k > 0:
            n_active = (out > 0).sum(dim=-1)
            if (n_active > self.lateral_k).any():
                kth_val, _ = torch.kthvalue(
                    membrane, membrane.shape[-1] - self.lateral_k, dim=-1
                )
                inhibition_mask = (membrane >= kth_val.unsqueeze(-1)).float()
                out = out * inhibition_mask

        return out.squeeze(0) if sq else out


    @torch.no_grad()
    def update_traces(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor
    ) -> Dict[str, float]:
        """STDP iz güncellemesi ve eligibility birikimi."""
        self._step_count += 1
        if pre_spikes.dim() > 1:
            pre_spikes = pre_spikes.mean(0)
        if post_spikes.dim() > 1:
            post_spikes = post_spikes.mean(0)

        self.trace_pre.mul_(self._decay_pre).add_(pre_spikes)
        self.trace_post.mul_(self._decay_post).add_(post_spikes)

        # Soft-bounded STDP
        soft_ltp = (self.w_max - self.weights.data).clamp_(min=0.0)
        soft_ltd = (self.weights.data - self.w_min).clamp_(min=0.0)
        raw_ltp = torch.outer(self.trace_pre, post_spikes)
        raw_ltd = torch.outer(pre_spikes, self.trace_post)
        stdp_signal = self.A_ltp * raw_ltp * soft_ltp - self.A_ltd * raw_ltd * soft_ltd

        self._last_ltp_mag = (self.A_ltp * raw_ltp * soft_ltp).abs().mean().item()
        self._last_ltd_mag = (self.A_ltd * raw_ltd * soft_ltd).abs().mean().item()

        self.eligibility.mul_(self._decay_elig).add_(stdp_signal)

        self.firing_rate_ema.mul_(self.ema_decay).add_(
            (1.0 - self.ema_decay) * post_spikes
        )
        self._last_sparsity = 1.0 - (
            (pre_spikes.sum() + post_spikes.sum()).item() / (self.n_in + self.n_out)
        )

        return {
            'ltp': self._last_ltp_mag,
            'ltd': self._last_ltd_mag,
            'elig': self.eligibility.abs().sum().item(),
        }

    @torch.no_grad()
    def apply_dopamine(self, rpe: float) -> torch.Tensor:
        """Dopamin sinyali ile eligibility → ağırlık güncellemesi."""
        self._da_event_count += 1
        self._last_rpe = rpe

        if abs(rpe) < self.dopamine_threshold:
            return torch.zeros_like(self.weights.data)

        delta_w = self.dopamine.modulate_gradients(
            gradients=self.eligibility, rpe=rpe
        )
        self.weights.data.add_(delta_w)
        self._last_delta_w = delta_w.abs().mean().item()
        self.eligibility.mul_(0.1)
        self._homeostatic_normalization()
        return delta_w

    @torch.no_grad()
    def _homeostatic_normalization(self) -> None:
        """Üç mekanizmalı homeostatik regülasyon."""
        W = self.weights.data
        W.clamp_(min=self.w_min, max=self.w_max)

        col_sums = W.sum(dim=0)
        over = col_sums > self.target_input_sum
        if over.any():
            ideal = self.target_input_sum / col_sums.clamp(min=1e-8)
            scale = torch.where(
                over,
                1.0 + self.scaling_speed * (ideal - 1.0),
                torch.ones_like(col_sums),
            )
            scale.clamp_(min=0.5, max=1.5)
            W.mul_(scale.unsqueeze(0))

        if self.firing_rate_ema.sum() > 0:
            rr = (self.target_firing_rate / (
                self.firing_rate_ema + 1e-8
            )).clamp_(0.85, 1.15)
            W.mul_((1.0 + self.scaling_speed * (rr - 1.0)).unsqueeze(0))

        W.clamp_(min=self.w_min, max=self.w_max)

    def get_telemetry(self) -> Dict[str, float]:
        W = self.weights.data
        return {
            'step': self._step_count,
            'da_events': self._da_event_count,
            'rpe': self._last_rpe,
            'ltp': self._last_ltp_mag,
            'ltd': self._last_ltd_mag,
            'delta_w': self._last_delta_w,
            'w_mean': W.mean().item(),
            'w_std': W.std().item(),
            'w_sparsity': (W < 0.01).float().mean().item(),
            'elig_energy': self.eligibility.abs().sum().item(),
            'rate_mean': self.firing_rate_ema.mean().item(),
            'spike_sparsity': self._last_sparsity,
        }

    def reset_traces(self) -> None:
        self.trace_pre.zero_()
        self.trace_post.zero_()
        self.eligibility.zero_()
        self.firing_rate_ema.zero_()

    @torch.no_grad()
    def som_update(
        self,
        pre_spikes: torch.Tensor,
        learning_rate: float = 0.05,
        neighborhood_decay: float = 0.5,
    ) -> None:
        """
        Kohonen SOM (Self-Organizing Map) agirlik guncelleme kurali.

        Biology: Kortekste uzamsal topolojiyi olusturan mekanizmanin
        hesaplamali modeli. En iyi eslesme yapan noron (BMU) ve fiziksel
        2D grid komsuları girdiye dogru cekılir. Boylece benzer girdiler
        zamanla grid uzerinde yakin noktalara kume yapar.

        STDP ile farki:
          - STDP: "birlikte ates ediyorsa baglantıyı guclendir" (zamansal)
          - SOM : "en iyi eslesme yapan ve komsuları girdiye dogru cekil" (uzamsal)

        Her ikisi de paralel calisir:
          1. update_traces() + apply_dopamine() -> STDP/Hebbian ogrenme
          2. som_update()                       -> topolojik organizasyon

        Cagri zamani: forward() sonrasinda, her ogrenme adımında.
        Sadece topology_grid != None olan katmanlarda (L23) aktif.

        Args:
            pre_spikes:        Giris vektoru (n_in,) - L4 cikisi
            learning_rate:     SOM ogrenme hizi (lr). Genellikle 0.01-0.1.
            neighborhood_decay: Komsu etkisi azalma katsayisi.
                               Kucuk sigma_eff -> dar komşuluk -> ince ayar.
        """
        if not self._has_topology:
            return  # Sadece L23 (topology_grid != None) icin calis

        if pre_spikes.dim() > 1:
            pre_spikes = pre_spikes.squeeze(0)
        if pre_spikes.shape[0] != self.n_in:
            return  # Shape uyumsuzlugu - sessizce atla

        W = self.weights.data  # (n_in, n_out)
        H = self._grid_H
        W_grid = self._grid_W

        # ── Adim 1: BMU (Best Matching Unit) bul ────────────────────
        # Her cikis noronunun agirlik vektoru ile giris arasindaki mesafe
        # Cosine benzerligi yerine dot product kullan (spike-based icin uygun)
        scores = torch.mv(W.t(), pre_spikes)  # (n_out,)
        bmu_idx = torch.argmax(scores).item()
        bmu_row = bmu_idx // W_grid
        bmu_col = bmu_idx % W_grid

        # ── Adim 2: Her noronun BMU'ya olan grid mesafesini hesapla ─
        # Vektorize: tum noronlarin (row, col) koordinatları
        # Device uyusmazligi hatasini onlemek icin self.weights.device uzerinde olustur
        device = self.weights.device
        rows = torch.arange(H, dtype=torch.float32, device=device)    # (H,)
        cols = torch.arange(W_grid, dtype=torch.float32, device=device)  # (W,)
        grid_rows = rows.unsqueeze(1).expand(H, W_grid).reshape(-1)  # (N,)
        grid_cols = cols.unsqueeze(0).expand(H, W_grid).reshape(-1)  # (N,)

        dist_sq = (grid_rows - bmu_row) ** 2 + (grid_cols - bmu_col) ** 2  # (N,)

        # ── Adim 3: Komşuluk fonksiyonu (Gaussian) ──────────────────
        # sigma_eff: komşuluk yariçapi (grid adimi biriminde)
        # Kucuk sigma -> sadece BMU etkilenir (ince detay)
        # Buyuk sigma -> genis komşuluk (global organizasyon)
        sigma_eff = 2.0  # self.lat_sigma_exc ile eslesen deger
        neighborhood = torch.exp(-dist_sq / (2.0 * sigma_eff ** 2))  # (N,)

        # ── Adim 4: Agirlik guncelle ─────────────────────────────────
        # delta_W[:, j] = lr * h(j, BMU) * (pre_spikes - W[:, j])
        # "Noron j'nin agirligini girdiye dogru cek, katsayi komşuluğa gore azal"
        delta = pre_spikes.unsqueeze(1) - W           # (n_in, n_out): giris - agirlik
        update = learning_rate * neighborhood * delta  # (n_in, n_out): olcekli guncelleme
        W.add_(update)

        # Agirlik sinirlamalari koru
        W.clamp_(min=self.w_min, max=self.w_max)


    def __repr__(self) -> str:
        W = self.weights.data
        return (
            f"CorticalLayer({self.n_in}->{self.n_out}, "
            f"{self.n_in * self.n_out:,} synapses, "
            f"W={W.mean():.3f}±{W.std():.3f})"
        )


# ═══════════════════════════════════════════════════════════════════
#  CorticalColumn — Çok Katmanlı Kortikal İşlemci
#  Drop-in replacement for HybridHebbianLearner
# ═══════════════════════════════════════════════════════════════════

class CorticalColumn(nn.Module):
    """
    4 Katmanlı Kortikal Kolon — HybridHebbianLearner'ın çok katmanlı
    halefi.

    Aynı public API:
        forward(pre_spikes) → post_spikes
        update_traces(pre, post) → stats
        apply_dopamine(reward) → delta_w
        spreading_activation(intent) → intent
        learning_step(pre, post, reward) → telemetry
        get_telemetry() → dict
        reset_traces()
        reset_all()

    Aynı property'ler:
        weights, n_pre, n_post, n_hidden
        eligibility, trace_pre, trace_post, firing_rate_ema
        device, lateral_k
        _step_count, _da_event_count

    Katman yapısı:
        L4  : n_pre   → n_hidden  (giriş, özellik çıkarımı)
        L23 : n_hidden → n_hidden  (entegrasyon, lateral bağlantılar)
        L5  : n_hidden → n_post   (çıkış, neural_intent)
        L6  : [pasif]             (Faz 3: prediktif geri besleme)
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        n_hidden: int = 1024,
        # STDP parametreleri (tüm katmanlara aktarılır)
        dt: float = 1.0,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        tau_eligibility: float = 25.0,
        A_ltp: float = 0.005,
        A_ltd: float = 0.003,
        w_max: float = 1.0,
        w_min: float = 0.0,
        target_input_sum: float = 40.0,
        scaling_speed: float = 0.001,
        target_firing_rate: float = 0.1,
        dopamine_threshold: float = 0.01,
        spike_threshold: float = 1.0,
        lateral_k: int = 10,
        ema_decay: float = 0.99,
        # Gamma/critic (merkezi dopamin modülü için)
        gamma: float = 0.99,
        lr_critic: float = 0.1,
        device: str = 'cpu',
    ):
        super().__init__()

        self._n_pre = n_pre
        self._n_post = n_post
        self._n_hidden = n_hidden
        self.device = device
        self.lateral_k = lateral_k

        # Ortak parametre seti (her katmana aktarılır)
        layer_kwargs = dict(
            dt=dt,
            tau_pre=tau_pre,
            tau_post=tau_post,
            tau_eligibility=tau_eligibility,
            A_ltp=A_ltp,
            A_ltd=A_ltd,
            w_max=w_max,
            w_min=w_min,
            target_input_sum=target_input_sum,
            scaling_speed=scaling_speed,
            target_firing_rate=target_firing_rate,
            dopamine_threshold=dopamine_threshold,
            spike_threshold=spike_threshold,
            ema_decay=ema_decay,
            device=device,
        )

        # ── Katman tanımları ──
        # L4: Thalamik giriş (Wernicke spike'ları buraya gelir)
        # Topoloji yok — düz özellik çıkarımı katmanı
        self.L4 = CorticalLayer(
            n_in=n_pre,
            n_out=n_hidden,
            lateral_k=lateral_k,
            mask_first_n=0,
            topology_grid=None,
            **layer_kwargs,
        )

        # L23: İçsel entegrasyon ve topolojik yanal bağlantılar
        # ─────────────────────────────────────────────────────
        # Biyolojik karşılık: Supragranüler katman (L2/3) kortikal kolonlar arası
        # entegrasyonu sağlar. Mexican Hat yanal bağlantılarla topolojik bir
        # anlamsal harita (semantic map) oluşturur.
        #
        # Parametreler:
        #   topology_grid=(32, 32) → n_hidden=1024 nöron, 32×32 grid
        #   sigma_exc=2.0  → ~8-12 yakın komşu uyarılır
        #   sigma_inh=5.0  → daha geniş halka inhibe edilir
        #   k-WTA(50) + Mexican Hat birlikte çalışır:
        #     Mexican Hat sigma_exc küçük olduğu için <50 nöron aktifleşir,
        #     k-WTA devreye girmez → topoloji korunur.
        _l23_H, _l23_W = 32, 32  # n_hidden = 1024 = 32*32
        assert n_hidden == _l23_H * _l23_W, (
            f"L23 topoloji için n_hidden={n_hidden} olmalı {_l23_H}*{_l23_W}=1024. "
            f"Farklı n_hidden kullanmak istiyorsanız topology_grid parametresini güncelleyin."
        )
        self.L23 = CorticalLayer(
            n_in=n_hidden,
            n_out=n_hidden,
            lateral_k=lateral_k,
            mask_first_n=0,
            topology_grid=(_l23_H, _l23_W),
            lat_sigma_exc=2.0,
            lat_sigma_inh=5.0,
            lat_A_exc=0.5,
            lat_A_inh=0.25,
            lat_spectral_target=0.9,
            **layer_kwargs,
        )

        # L5: Çıkış katmanı — neural_intent buradan üretilir
        # Özel token maskesi sadece burada aktif (HybridHebbianLearner ile aynı)
        self.L5 = CorticalLayer(
            n_in=n_hidden,
            n_out=n_post,
            lateral_k=lateral_k,
            mask_first_n=NUM_MASKED_TOKENS,
            **layer_kwargs,
        )

        # L6: Multiformis katmanı (Tahmin / Prediktif Geri Besleme)
        # Bilişsel çıktıdan/bağlamdan (n_hidden) duyu girişine (n_pre) projeksiyon
        # FAZ 3: k-WTA inhibisyonu yoktur (tüm tahmin spektrumu korunmalıdır)
        self.L6 = CorticalLayer(
            n_in=n_hidden,
            n_out=n_pre,
            lateral_k=0,
            mask_first_n=0,
            **layer_kwargs,
        )

        # Prediktif geri besleme kontrolü ve bellek hücresi
        self.predictive_feedback = False
        self._prev_h23 = torch.zeros(n_hidden, device=device)

        # Merkezi dopamin modülü (tüm katmanlar için tek RPE hesabı)
        self._dopamine = DopamineModulator(gamma=gamma, lr_critic=lr_critic)

        # Telemetri (uyumluluk için)
        self._step_count = 0
        self._da_event_count = 0
        self._last_rpe = 0.0
        self._last_ltp_mag = 0.0
        self._last_ltd_mag = 0.0
        self._last_delta_w = 0.0
        self._last_sparsity = 0.0

    # ─────────────────────────────────────────────────────────────────
    #  HybridHebbianLearner UYUMLULUK PROPERTY'LERİ
    # ─────────────────────────────────────────────────────────────────

    @property
    def weights(self) -> nn.Parameter:
        """Dışarıdan 'weights' erişimi → L5 çıkış katmanının ağırlıkları.
        .mx save/load, spreading_activation ve hebbian_trace erişimi
        bu property üzerinden çalışır."""
        return self.L5.weights

    @property
    def n_pre(self) -> int:
        return self._n_pre

    @property
    def n_post(self) -> int:
        return self._n_post

    @property
    def n_hidden(self) -> int:
        return self._n_hidden

    @property
    def eligibility(self) -> torch.Tensor:
        """L5 eligibility trace (uyumluluk için)."""
        return self.L5.eligibility

    @eligibility.setter
    def eligibility(self, value: torch.Tensor) -> None:
        self.L5.eligibility = value

    @property
    def trace_pre(self) -> torch.Tensor:
        """L4 pre-synaptic trace (uyumluluk için)."""
        return self.L4.trace_pre

    @trace_pre.setter
    def trace_pre(self, value: torch.Tensor) -> None:
        self.L4.trace_pre = value

    @property
    def trace_post(self) -> torch.Tensor:
        """L5 post-synaptic trace (uyumluluk için)."""
        return self.L5.trace_post

    @trace_post.setter
    def trace_post(self, value: torch.Tensor) -> None:
        self.L5.trace_post = value

    @property
    def firing_rate_ema(self) -> torch.Tensor:
        """L5 firing rate EMA (uyumluluk için)."""
        return self.L5.firing_rate_ema

    @firing_rate_ema.setter
    def firing_rate_ema(self, value: torch.Tensor) -> None:
        self.L5.firing_rate_ema = value

    # ─────────────────────────────────────────────────────────────────
    #  FORWARD PASS
    # ─────────────────────────────────────────────────────────────────

    def forward(self, pre_spikes: torch.Tensor, spiking: bool = True) -> torch.Tensor:
        """
        L4 -> L23 -> L5 tam ileri besleme (ve L6 prediktif geri besleme).
        HybridHebbianLearner.forward() ile aynı imza.

        Args:
            pre_spikes: (n_pre,) veya (batch, n_pre) tensör
            spiking: True ise L5 spiking çıktı üretir, False ise graded analog çıktı üretir.

        Returns:
            post_spikes: (n_post,) veya (batch, n_post) tensör
        """
        # ── Faz 3: Prediktif Tahmin Hatası (Predictive Coding) ──
        if self.predictive_feedback:
            prev_state = self._prev_h23
            if pre_spikes.dim() == 2:
                prev_state = prev_state.unsqueeze(0).expand(pre_spikes.shape[0], -1)
            
            # Bir önceki internal durumdan (L23) şimdiki girdiyi tahmin et
            prediction = self.L6.forward(prev_state, spiking=True)
            # Tahmin hatasını hesapla ve Thalamus'a ilet
            error_spikes = (pre_spikes - prediction).clamp(min=0.0)
        else:
            error_spikes = pre_spikes

        # İç katmanlar (L4, L23) her zaman spike üretir
        h4  = self.L4.forward(error_spikes, spiking=True)
        h23 = self.L23.forward(h4, spiking=True)
        
        # Sonraki adım tahmini için L23 durumunu kaydet
        if pre_spikes.dim() == 1:
            self._prev_h23 = h23.clone()
        else:
            self._prev_h23 = h23.mean(0).clone() # Batch ortalamasını al
            
        # Çıkış katmanı (L5) spiking veya graded olabilir
        out = self.L5.forward(h23, spiking=spiking)
        return out


    # ─────────────────────────────────────────────────────────────────
    #  ÖĞRENME PIPELINE
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_traces(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor
    ) -> Dict[str, float]:
        """
        Tüm katmanların STDP izlerini güncelle.

        İleri geçişin ara çıktılarını yeniden hesaplayarak her katmanın
        pre/post çiftini doğru şekilde günceller.
        """
        self._step_count += 1

        # Ara aktivasyonları yeniden hesapla (no_grad altında ucuz)
        if self.predictive_feedback:
            prev_state = self._prev_h23
            if pre_spikes.dim() == 2:
                prev_state = prev_state.unsqueeze(0).expand(pre_spikes.shape[0], -1)
            prediction = self.L6.forward(prev_state, spiking=True)
            error_spikes = (pre_spikes - prediction).clamp(min=0.0)
        else:
            error_spikes = pre_spikes

        h4  = self.L4.forward(error_spikes)
        h23 = self.L23.forward(h4)

        # Her katman kendi pre→post izini günceller
        stats_L4  = self.L4.update_traces(error_spikes, h4)
        stats_L23 = self.L23.update_traces(h4, h23)
        stats_L5  = self.L5.update_traces(h23, post_spikes)
        
        # L6, iç temsil (h23) ile gelen ham girdi (pre_spikes) arasındaki tahmini öğrenir
        stats_L6  = self.L6.update_traces(h23, pre_spikes)

        # Sonraki adım tahmini için L23 durumunu kaydet
        if pre_spikes.dim() == 1:
            self._prev_h23 = h23.clone()
        else:
            self._prev_h23 = h23.mean(0).clone()

        # Üst-düzey telemetri (L5 baskın, uyumluluk için)
        self._last_ltp_mag = stats_L5['ltp']
        self._last_ltd_mag = stats_L5['ltd']
        self._last_sparsity = self.L5._last_sparsity

        return {
            'ltp': self._last_ltp_mag,
            'ltd': self._last_ltd_mag,
            'elig': (
                stats_L4['elig'] + stats_L23['elig'] + stats_L5['elig'] + stats_L6['elig']
            ),
        }

    @torch.no_grad()
    def apply_dopamine(
        self, reward: float, new_value_estimate: Optional[float] = None
    ) -> torch.Tensor:
        """
        Tek RPE hesabı → tüm katmanlara yayılır.
        HybridHebbianLearner.apply_dopamine() ile aynı imza.
        """
        self._da_event_count += 1
        rpe = self._dopamine.compute_rpe(
            reward=reward, new_value_estimate=new_value_estimate
        )
        self._last_rpe = rpe

        # Her katmanın eligibility × rpe → ağırlık güncellemesi
        dw_L4  = self.L4.apply_dopamine(rpe)
        dw_L23 = self.L23.apply_dopamine(rpe)
        dw_L5  = self.L5.apply_dopamine(rpe)
        dw_L6  = self.L6.apply_dopamine(rpe)

        self._last_delta_w = (
            dw_L4.abs().mean() + dw_L23.abs().mean() + dw_L5.abs().mean() + dw_L6.abs().mean()
        ).item() / 4.0

        # L5 delta_w'yi döndür (uyumluluk)
        return dw_L5

    def learning_step(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: float,
        new_value_estimate: Optional[float] = None,
        som_lr: float = 0.0,
    ) -> Dict[str, float]:
        """
        Tam ogrenme adimi: STDP trace + Dopamin + (opsiyonel) L23 SOM guncelleme.

        Args:
            pre_spikes:          L4 giris spike vektoru (n_pre,)
            post_spikes:         L5 cikis spike vektoru (n_post,)
            reward:              Odul sinyali (RPE hesabi icin)
            new_value_estimate:  Yeni deger tahmini (temporal fark icin, opsiyonel)
            som_lr:              L23 SOM ogrenme hizi. 0.0 (varsayilan) -> SOM pasif.
                                 Egitim sirasinda 0.05-0.3 araliginda kullanilir.
                                 Canlı sistem (Limbic/Dream) 0.0 kullanir.
        """
        # Adim 1: STDP izleri + dopamin (tum katmanlar)
        self.update_traces(pre_spikes, post_spikes)
        self.apply_dopamine(reward, new_value_estimate)

        # Adim 2: L23 SOM topolojik guncelleme (opsiyonel, sadece egitimde)
        # pre_spikes L4 girisine karsili L23'un girdisi h4'tur.
        # h4'u yeniden hesaplamak yerine L4'un son cikisini kullaniyoruz.
        if som_lr > 0.0:
            with torch.no_grad():
                h4 = self.L4.forward(pre_spikes, spiking=True)
                self.L23.som_update(h4, learning_rate=som_lr)

        return self.get_telemetry()


    # ─────────────────────────────────────────────────────────────────
    #  YAYILIMSAL AKTİVASYON (HybridHebbianLearner ile aynı)
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def spreading_activation(
        self,
        initial_activation: torch.Tensor,
        steps: int = 3,
        alpha: float = 0.85,
    ) -> torch.Tensor:
        """
        L5 ağırlıkları üzerinden kavram yayılımı.
        HybridHebbianLearner.spreading_activation() ile tamamen aynı.
        """
        W = self.L5.weights.data

        norms = torch.norm(W, p=2, dim=0, keepdim=True).clamp_min(1e-8)
        W_norm = W / norms

        T = torch.matmul(W_norm.t(), W_norm)
        T = torch.relu(T)
        T.fill_diagonal_(0.0)

        row_sums = T.sum(dim=1, keepdim=True).clamp_min(1e-8)
        T_trans = T / row_sums

        x = initial_activation.clone()
        x0 = initial_activation.clone()

        for _ in range(steps):
            x = alpha * torch.matmul(T_trans.t(), x) + (1 - alpha) * x0

        return x

    # ─────────────────────────────────────────────────────────────────
    #  TELEMETRİ VE SIFIRLAMA
    # ─────────────────────────────────────────────────────────────────

    def get_telemetry(self) -> Dict[str, float]:
        """Tüm katmanların birleşik telemetrisi."""
        t5 = self.L5.get_telemetry()
        return {
            'step':          self._step_count,
            'da_events':     self._da_event_count,
            'rpe':           self._last_rpe,
            'ltp':           self._last_ltp_mag,
            'ltd':           self._last_ltd_mag,
            'delta_w':       self._last_delta_w,
            # L5 ağırlık istatistikleri (uyumluluk için)
            'w_mean':        t5['w_mean'],
            'w_std':         t5['w_std'],
            'w_sparsity':    t5['w_sparsity'],
            'elig_energy':   t5['elig_energy'],
            'rate_mean':     t5['rate_mean'],
            'spike_sparsity': t5['spike_sparsity'],
            # Ek: katman bilgisi
            'n_layers':      3,
            'n_hidden':      self._n_hidden,
        }

    def reset_traces(self) -> None:
        """Tüm katmanların izlerini sıfırla."""
        self.L4.reset_traces()
        self.L23.reset_traces()
        self.L5.reset_traces()
        self.L6.reset_traces()
        self._dopamine.value_estimate = 0.0
        # Reset prediction context
        if self._prev_h23 is not None:
            self._prev_h23.zero_()

    def reset_all(self) -> None:
        """Tüm izler + tüm ağırlıklar (fresh init)."""
        self.reset_traces()
        with torch.no_grad():
            self.L4.weights.data.uniform_(0, 0.3)
            self.L23.weights.data.uniform_(0, 0.3)
            self.L5.weights.data.uniform_(0, 0.3)
            self.L6.weights.data.uniform_(0, 0.05) # L6 da sıfırlanır (küçük ağırlıklar)
        self._step_count = 0
        self._da_event_count = 0

    def __repr__(self) -> str:
        total_synapses = (
            self._n_pre * self._n_hidden
            + self._n_hidden * self._n_hidden
            + self._n_hidden * self._n_post
        )
        return (
            f"CorticalColumn(\n"
            f"  L4 : {self.L4}\n"
            f"  L23: {self.L23}\n"
            f"  L5 : {self.L5}\n"
            f"  Total synapses: {total_synapses:,}\n"
            f"  Steps: {self._step_count}, DA events: {self._da_event_count}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════
#  HIZLI TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("CorticalColumn — Hızlı Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    col = CorticalColumn(n_pre=768, n_post=1416, n_hidden=1024, device=device)
    print(f"\n{col}\n")

    # Forward test
    pre = torch.zeros(768, device=device)
    pre[:20] = (torch.rand(20, device=device) < 0.3).float()
    post = col.forward(pre)
    print(f"Forward OK — post shape: {post.shape}, sum: {post.sum().item():.2f}")

    # update_traces test
    col.update_traces(pre, post)
    print(f"update_traces OK")

    # apply_dopamine test
    dw = col.apply_dopamine(reward=1.0)
    print(f"apply_dopamine OK — dw L5 mean: {dw.abs().mean().item():.6f}")

    # spreading_activation test
    intent = torch.zeros(1416, device=device)
    intent[10] = 1.0
    spread = col.spreading_activation(intent)
    print(f"spreading_activation OK — spread sum: {spread.sum().item():.4f}")

    # Uyumluluk kontrolleri
    print(f"\nUyumluluk kontrolleri:")
    print(f"  weights.shape:       {col.weights.shape}")
    print(f"  n_pre:               {col.n_pre}")
    print(f"  n_post:              {col.n_post}")
    print(f"  eligibility.shape:   {col.eligibility.shape}")
    print(f"  trace_pre.shape:     {col.trace_pre.shape}")
    print(f"  trace_post.shape:    {col.trace_post.shape}")
    print(f"  firing_rate_ema.shape: {col.firing_rate_ema.shape}")
    print(f"  device:              {col.device}")
    print(f"  lateral_k:           {col.lateral_k}")

    tele = col.get_telemetry()
    print(f"\nTelemetri: {tele}")
    print("\n" + "=" * 60)
    print("Tüm testler GECTI")
    print("=" * 60)
