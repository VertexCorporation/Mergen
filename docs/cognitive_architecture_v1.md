# Mergen Bilişsel Mimari Geliştirme Raporu (Sürüm 1.0)

Bu belge, Mergen'in yapay sinir ağı tabanlı çekirdeğini reaktif bir sistemden **proaktif, biyofiziksel olarak modellenmiş, nöromodülatör kontrollü bir "yapay beyne"** dönüştürmek amacıyla yapılan Faz 1, Faz 2 ve Faz 3 geliştirmelerini özetlemektedir.

---

## Faz 1: Nöromodülasyon Altyapısı ve Güvenli Refactoring

İlk aşamada spagetti kod yapısı kırılarak modüler, katmanlı bir hiyerarşiye (Layered Architecture) geçilmiştir. Ayrıca sistemin tüm biyokimyasını yönetecek temel altyapı atılmıştır.

* **Dizin Yapılanması:** Kod tabanı `core/`, `cognitive/` ve `memory/` olarak mantıksal modüllere ayrılmıştır.
* **Nöromodülatör Sistemi (`NeuromodulationSystem`):** 
  Sistemin genel duygu ve öğrenme durumunu kontrol eden 4 ana tensör inşa edilmiştir:
  * **Dopamin (DA):** Ödül, motivasyon, öğrenme hızı (LTP).
  * **Serotonin (5-HT):** İnhibisyon (baskılama), sabır.
  * **Noradrenalin (NE):** Dikkat (odak daralması), stres, uyanıklık.
  * **Asetilkolin (ACh):** Hafıza kodlama gücü, yeniliğe duyarlılık.
* **GPU ve Bellek Güvenliği:** Tüm nöromodülatör ve bellek tensörleri, PyTorch'un otomatik gradyan takibinden (autograd) çıkarılmış (`requires_grad=False`) ve in-place operasyonlar kullanılarak (örn. `tensor.add_()` , `tensor.clamp_()`) **memory leak** (bellek kaçağı) riskleri tamamen ortadan kaldırılmıştır.
* **Serialization (Kayıt) Güvenliği:** Tensörlerin diske standart `float` veri tiplerinde yazılması sağlanmış, eski kayıt dosyalarında nöromodülatörler olmasa bile sistemin çökmesini engelleyen güvenli Fallback (Geriye dönük uyumluluk) mekanizması yazılmıştır.

---

## Faz 2: Talamik Süzgeç ve Global İnhibisyon

Sistemin "Girdi (Duyu)" ve "Çıktı (İfade)" aşamalarına iki kritik biyolojik kalkan yerleştirilmiştir.

* **Talamus Filtresi (`ThalamicFilter`):** 
  Wernicke alanından (veya RAG'dan) çıkan kavramsal veriler, üst kortekse (Hebbian'a) gitmeden önce Talamus süzgecinden geçirilir. Bu süzgeç dinamiktir:
  * **NE (Noradrenalin) yüksekse:** Eşik (threshold) artar. Sadece çok güçlü ve önemli sinyaller içeri girer (Stres altındayken sadece hedefe odaklanma).
  * **DA (Dopamin) yüksekse:** Eşik düşer. En ufak sinyaller ve detaylar bile algılanır (Merak ve geniş algı).
  * *Teknik Detay:* Filtreleme işlemi `torch.where` ile in-place maskeleme yapılarak optimize edilmiştir.
* **Serotonin İnhibisyonu (Halüsinasyon Baskılama):**
  Limbic sistem yanıt üretirken, yanıtın "Güven Skoru" (`activation_strength`) hesaplanır. Eğer bu güven skoru **Serotonin (5-HT)** inhibisyon eşiğinden düşükse, sistem uydurmak (hallucination) yerine kendi Broca alanını by-pass edip anında: *"Bu konuda net bir fikrim yok, uydurmak istemiyorum"* yanıtını verir.

---

## Faz 3: Çalışma Belleği (Prefrontal Korteks)

Mergen'in sadece anlık tepki veren değil, kısa süreli bağlamı aklında tutabilen bir Çalışma Belleği (`WorkingMemory`) inşa edilmiştir.

* **Tensör Tabanlı Slot Havuzu:** Sınırlı kapasiteye sahip (örn. 5 slot), `n_pre` boyutunda konsept vektörlerini tutan in-place bir hafıza modülü yazılmıştır.
* **ACh ile Kodlama:** Yeni gelen bir bilginin hafızaya yazılma gücü (başlangıç aktivasyonu) Asetilkolin (ACh) tarafından belirlenir. ACh çok düşükse bilgi hafızaya yazılmaz bile.
* **NE ile Tutunma (Decay):** Hafızadaki bilgiler zamanla (`tick()`) unutulur (aktivasyonları düşer). Ancak NE (Noradrenalin / Odak) yüksekse bu unutma yavaşlar, bilgi uzun süre taze kalır.
* **Aktivasyon Bazlı Eviction:** Bellek dolduğunda yeni bilgi geldiğinde eski FIFO (ilk giren çıkar) kuralı yerine, **aktivasyonu en düşük olan** (yani dikkatin en çok dağıldığı) bilginin üzerine yazılır. Bu bilişsel gerçekliğe uygun bir yaklaşımdır.

---

## Faz 3.2: Predictive Processing (Beklenti Üretimi)

Sistemin pasif dinleyicilikten çıkarak "bir sonraki adımı tahmin etmesi" ve yanıldığında öğrenmesini sağlayan nöro-bilişsel mekanizma eklenmiştir.

* **Beklenti (Prediction) Oluşturma:** Çalışma belleğinde aktif olan konseptlerin **ağırlıklı ortalaması (weighted average)** alınarak `predicted_next_vector` üretilir. Sistem bir sonraki saniye duyacağı şeyin, aklındakilere benzeyeceğini tahmin eder.
* **Şaşkınlık (Surprise) Hesabı:** Yeni bir girdi geldiğinde, bu beklenti vektörü ile arasındaki **Cosine Similarity (Kosinüs Benzerliği)** hesaplanır. Girdi beklentiden ne kadar uzaksa `Surprise` o kadar yüksek olur. (*NaN ve Sıfıra bölünme hatalarına karşı Epsilon koruması eklenmiştir*).
* **Dopamin (DA) Sıçraması:** Eğer Surprise (Şaşkınlık) belli bir eşiğin üzerindeyse, sistem anlık olarak Dopamin salgılar (`da_delta`).
* **Sonuç (LTP Tetiklenmesi):** Bu dopamin sıçraması, Hebbian ağlarındaki öğrenme hızını artırır. Yani Mergen **şaşırdığı ve beklemediği olaylardan çok daha hızlı ve güçlü bir şekilde öğrenir.**

---

*Tüm bu geliştirmeler, 10'dan fazla izole birim testiyle (`tests/` klasörü altında) doğrulanmış olup, GPU bellek dostu (memory-leak free) kurallarına %100 sadık kalınarak başarıyla entegre edilmiştir.*
