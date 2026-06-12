"""
Eğitim Müfredatı - 1. sınıftan 12. sınıfa kadar konular
"""

CURRICULUM = {
    1: {
        "name": "1. Sınıf",
        "subjects": {
            "Türkçe": ["Sesler", "Harfler", "Basit kelimeler", "Cümle kurma"],
            "Matematik": ["Sayılar 1-20", "Toplama", "Çıkarma", "Şekiller"],
            "Hayat Bilgisi": ["Aile", "Okul", "Duyular", "Güvenlik"],
        }
    },
    2: {
        "name": "2. Sınıf",
        "subjects": {
            "Türkçe": ["Okuma", "Yazma", "Hikaye", "Şiir"],
            "Matematik": ["Sayılar 1-100", "Toplama/Çıkarma", "Ölçüler", "Zaman"],
            "Hayat Bilgisi": ["Sağlık", "Çevre", "İletişim", "Kurallar"],
        }
    },
    3: {
        "name": "3. Sınıf",
        "subjects": {
            "Türkçe": ["Metin analizi", "Sözcük türleri", "Noktalama", "Anlatım"],
            "Matematik": ["Dört işlem", "Kesirler", "Geometri", "Problemler"],
            "Fen Bilimleri": ["Canlılar", "Maddeler", "Işık-Ses", "Güneş-Dünya"],
            "Sosyal Bilgiler": ["Haklarım", "Atatürk", "Çevremiz", "Haritalar"],
        }
    },
    4: {
        "name": "4. Sınıf",
        "subjects": {
            "Türkçe": ["Paragraf", "Anlatım türleri", "Sözlü iletişim", "Görsel okuma"],
            "Matematik": ["Kesirler", "Çarpma bölme", "Üçgenlar", "Veri toplama"],
            "Fen Bilimleri": ["Kuvvet-hareket", "Maddenin halleri", "Döngüler", "Elektrik"],
            "Sosyal Bilgiler": ["Kültürümüz", "Ülkemiz", "Kaynaklar", "Yönetim"],
        }
    },
    5: {
        "name": "5. Sınıf",
        "subjects": {
            "Türkçe": ["Metin türleri", "Sözcükte anlam", "Cümlede anlam", "Anlatım bozuklukları"],
            "Matematik": ["Kesirler", "Ondalık", "Yüzdeler", "Geometri"],
            "Fen Bilimleri": ["Hücre", "Kuvvetler", "Maddenin yapısı", "Işığın yansıması"],
            "Sosyal Bilgiler": ["Tarih bilinci", "Coğrafi bölgeler", "Ekonomi", "Demokrasi"],
            "İngilizce": ["Alfabe", "Renkler", "Sayılar", "Tanışma"],
        }
    },
    6: {
        "name": "6. Sınıf",
        "subjects": {
            "Türkçe": ["Fiilimsiler", "Cümle çeşitleri", "Paragraf", "Sözcükte anlam"],
            "Matematik": ["Tam sayılar", "Rasyonel sayılar", "Açılar", "Veri analizi"],
            "Fen Bilimleri": ["Vücudumuz", "Kuvvet-hareket", "Maddenin tanecikli yapısı", "Elektrik"],
            "Sosyal Bilgiler": ["Tarihsel olaylar", "Harita bilgisi", "Kültür ve sanat", "Küresel bağlantılar"],
            "İngilizce": ["Günlük hayat", "Hobiler", "Okul hayatı", "Aile"],
        }
    },
    7: {
        "name": "7. Sınıf",
        "subjects": {
            "Türkçe": ["Sözcük türleri", "Cümle analizi", "Paragraf", "Anlatım bozuklukları"],
            "Matematik": ["Tam sayılar", "Rasyonel sayılar", "Oran-orantı", "Cebirsel ifadeler"],
            "Fen Bilimleri": ["Hücre bölünmesi", "Kuvvet ve enerji", "Maddenin yapısı", "Sistemler"],
            "Sosyal Bilgiler": ["Tarihsel süreç", "Coğrafi bölgeler", "Üretim-dağıtım", "Küresel sorunlar"],
            "İngilizce": ["Günlük konuşmalar", "Geçmiş zaman", "Gelecek planları", "Seyahat"],
        }
    },
    8: {
        "name": "8. Sınıf",
        "subjects": {
            "Türkçe": ["Sözcükte anlam", "Cümlede anlam", "Paragraf", "Anlatım bozuklukları"],
            "Matematik": ["Çarpanlar-katlar", "Üslü sayılar", "Kareköklü sayılar", "Cebir"],
            "Fen Bilimleri": ["Hücre", "Kuvvet-hareket", "Maddenin yapısı", "Işığın yapısı"],
            "T.C. İnkılap Tarihi": ["Birinci Dünya Savaşı", "Kurtuluş Savaşı", "Cumhuriyet", "Atatürk"],
            "İngilizce": ["Günlük hayat", "Sağlık", "Teknoloji", "Çevre"],
        }
    },
    9: {
        "name": "9. Sınıf",
        "subjects": {
            "Türk Dili ve Edebiyatı": ["Sözcükte anlam", "Cümlede anlam", "Paragraf", "Edebi türler"],
            "Matematik": ["Kümeler", "Sayılar", "Bağıntı-fonksiyon", "Polinomlar"],
            "Fizik": ["Fizik bilimine giriş", "Madde ve özellikleri", "Kuvvet-hareket", "Enerji"],
            "Kimya": ["Kimya bilimi", "Atom ve periyodik sistem", "Maddenin halleri", "Asitler-bazlar"],
            "Biyoloji": ["Canlılar dünyası", "Hücre", "Sistemler", "Genetik"],
            "Tarih": ["İslam öncesi Türk tarihi", "İlk Müslüman Türk devletleri", "Orta Asya"],
            "İngilizce": ["Günlük hayat", "Eğitim", "Sağlık", "Teknoloji"],
        }
    },
    10: {
        "name": "10. Sınıf",
        "subjects": {
            "Türk Dili ve Edebiyatı": ["İslamiyet etkisinde gelişen Türk edebiyatı", "Divan edebiyatı", "Halk edebiyatı"],
            "Matematik": ["Çember ve daire", "Analitik düzlem", "Fonksiyonlar", "Polinomlar"],
            "Fizik": ["Kuvvet ve hareket", "Enerji", "Isı ve sıcaklık", "Elektrostatik"],
            "Kimya": ["Kimyasal hesaplamalar", "Gazlar", "Sıvı çözeltiler", "Asitler-bazlar"],
            "Biyoloji": ["Hücre bölünmesi", "Kalıtım", "Ekosistemler", "Canlılık"],
            "Tarih": ["Selçuklu Devleti", "Anadolu'da Türk beylikleri", "Osmanlı Devleti kuruluş"],
            "İngilizce": ["Günlük hayat", "Eğitim", "Kariyer", "Toplum"],
        }
    },
    11: {
        "name": "11. Sınıf",
        "subjects": {
            "Türk Dili ve Edebiyatı": ["Batı etkisinde gelişen Türk edebiyatı", "Servet-i Fünun", "Milli edebiyat"],
            "Matematik": ["İkinci dereceden denklemler", "Logaritma", "Diziler", "Seriler"],
            "Fizik": ["Kuvvet ve hareket", "Enerji momentum", "Elektrik", "Manyetizma"],
            "Kimya": ["Kimyasal tepkimeler", "Karışımlar", "Asitler-bazlar-tuzlar", "Termokimya"],
            "Biyoloji": ["Hücre", "Kalıtım", "Ekosistem", "Canlılık"],
            "Tarih": ["Osmanlı Devleti yükselme", "Osmanlı Devleti duraklama", "Dünya gücü"],
            "İngilizce": ["Günlük hayat", "Eğitim", "Kariyer", "Küresel sorunlar"],
        }
    },
    12: {
        "name": "12. Sınıf",
        "subjects": {
            "Türk Dili ve Edebiyatı": ["Cumhuriyet dönemi Türk edebiyatı", "Çağdaş Türk edebiyatı", "Edebi akımlar"],
            "Matematik": ["Çember", "Parabol", "Türev", "İntegral"],
            "Fizik": ["Kuvvet ve hareket", "Enerji", "Dalgalar", "Optik"],
            "Kimya": ["Kimyasal tepkimeler", "Karbon kimyasına giriş", "Enerji kaynakları", "Çevre kimyası"],
            "Biyoloji": ["Hücre", "Kalıtım", "Ekosistem", "Canlılık"],
            "Tarih": ["20. yüzyılda Osmanlı", "Türkiye Cumhuriyeti", "Çağdaş Türkiye", "Dünya"],
            "İngilizce": ["Günlük hayat", "Eğitim", "Kariyer", "Küresel sorunlar"],
        }
    },
}

def get_all_topics():
    """Tüm müfredat konularını düz liste olarak döndür."""
    all_topics = []
    for grade, data in CURRICULUM.items():
        for subject, topics in data["subjects"].items():
            for topic in topics:
                all_topics.append(f"{grade}. Sınıf - {subject}: {topic}")
    return all_topics

def get_grade_topics(grade: int):
    """Belirli bir sınıfın konularını döndür."""
    if grade in CURRICULUM:
        return CURRICULUM[grade]
    return None
