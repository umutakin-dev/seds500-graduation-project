# BiggSinerji 2026/1 — Stage 1 İş Fikri Başvurusu

> **Durum:** TASLAK — gözden geçirilmesi ve PRODİS portalına girilmesi gerekir
> **Son Tarih:** 8 Nisan 2026, 00:00
> **Portal:** eteydeb.tubitak.gov.tr → Çağrı #54 "BIGGSINERJI 2026/1"
> **Değerlendirme:** Her kriter 10 üzerinden, minimum 7 gerekli

---

## Başvuru Kontrol Listesi

- [ ] PRODİS portalına kayıt/giriş (eteydeb.tubitak.gov.tr)
- [ ] Çağrı #54 "BIGGSINERJI 2026/1" bul
- [ ] "İş Fikri Formu Başvurusu Yap" tıkla
- [ ] Bölüm 1: Kişisel bilgiler doldur
- [ ] Bölüm 1: Girişimci Yetenek ve Birikimi metin alanı (2000 karakter)
- [ ] Bölüm 1: Eğitim belgesi yükle (PDF — diploma veya transkript)
- [ ] Bölüm 1: Özgeçmiş yükle (PDF)
- [ ] Bölüm 2: İş Fikri Adı gir
- [ ] Bölüm 2: İş Fikri Başvuru Dokümanı yükle (Word→PDF, maks 5MB)
- [ ] "Başvuruyu Gönder" butonuna BAS (sadece kaydetmek yetmez!)
- [ ] Taahhütname ve Ön Yazı oluştur, imzala, DEPARK'a posta ile gönder

---

## İş Fikri Adı

**SynthShield: Difüzyon Modelleri ile Gizlilik Korumalı Sentetik Tablo Verisi Üretim Platformu**

---

## Girişimci ve Ekibinin Yetenek ve Birikimi (2000 karakter)

> Bu metin PRODİS portalındaki metin alanına girilecektir.

Yapay zeka ve makine öğrenmesi alanında uzmanlaşmış bir yazılım mühendisiyim. İzmir Yüksek Teknoloji Enstitüsü'nde (İYTE) yakın zamanda tamamladığım yüksek lisans tezimde, difüzyon modelleri kullanarak gizlilik korumalı sentetik tablo verisi üretimi üzerine araştırma yaptım. Bu araştırma, doğrudan SynthShield iş fikrinin teknik temelini oluşturmaktadır.

Araştırma sürecinde 10 farklı veri seti üzerinde 40'tan fazla deney gerçekleştirdim, 4 farklı yöntemi karşılaştırdım ve ~15.600 satır kod yazdım. Sonuçlar, difüzyon tabanlı yaklaşımımızın gerçek verinin %87-99 kullanılabilirliğini korurken tam gizlilik sağladığını gösterdi (üyelik çıkarım saldırısı AUC=0.51, rastgele tahmine eşdeğer).

Profesyonel olarak NormDigital'de düzenlenmiş sektörler (ilaç, gümrük) için kurumsal yazılım geliştirme deneyimim bulunmaktadır. Bu deneyim, hem KVKK uyumluluk gereksinimlerini hem de kurumsal müşteri ihtiyaçlarını derinlemesine anlamamı sağlamıştır.

Teknik yetkinliklerim: Python, PyTorch, scikit-learn (ML); Flask/FastAPI, React (web); PostgreSQL, Docker (altyapı). Tam yığın geliştirme yapabilmem, MVP'yi sıfırdan tek başıma oluşturabilmem anlamına gelir.

Bu iş fikri bir konsept değil — tamamlanmış bir araştırma projesiyle desteklenen, deneysel olarak doğrulanmış bir teknolojidir. Küresel sentetik veri pazarı %35 CAGR ile büyürken Türkiye'de hiçbir yerli çözüm bulunmamaktadır. KVKK yaptırımlarının artması (2026'da ihlal başına 17M TL'ye kadar ceza) bu çözüme acil ihtiyaç olduğunu göstermektedir.

---

## İş Fikri Başvuru Dokümanı İçeriği

> Bu içerik Word şablonuna aktarılıp PDF olarak yüklenecektir.

### 1. İş Fikri Özeti

**SynthShield**, yapay zeka difüzyon modelleri kullanarak hassas tablo verilerinin gizlilik-güvenli sentetik kopyalarını üreten bir platformdur. Kuruluşlar hassas verilerini yükler, platform istatistiksel olarak eşdeğer ancak sıfır gerçek kayıt içeren sentetik bir veri seti üretir. Bu sentetik veri KVKK/GDPR riski olmadan serbestçe paylaşılabilir, ML modeli eğitiminde kullanılabilir ve araştırma için dağıtılabilir.

### 2. Problem

Sağlık, finans, üretim ve kamu sektöründeki kuruluşlar değerli tablo verisi toplar ancak bu veriyi paylaşamaz:

- **Yasal risk:** KVKK, kişisel ve hassas verilerin paylaşımını yasaklar. 2026'da ihlal başına 17 milyon TL'ye kadar ceza uygulanmaktadır.
- **Ticari sırlar:** Üretim parametreleri, fiyatlandırma stratejileri, maliyet yapıları paylaşılamaz.
- **Yeniden kimlik tespiti riski:** "Anonimleştirilmiş" veriler tersine mühendislikle çözülebilir.

Mevcut çözümler yetersizdir: anonimleştirme tersine çevrilebilir, veri maskeleme ML için kullanılabilirliği düşürür, GAN tabanlı yaklaşımlar (CTGAN) yalnızca %44-84 kullanılabilirlik sağlar.

### 3. Çözüm: Difüzyon Tabanlı Sentetik Veri

Difüzyon modelleri — DALL-E gibi görüntü üreticilerin arkasındaki teknoloji — tablo verisine uyarlanarak:

1. **İstatistiksel özellikleri korur:** Sentetik veri üzerinde eğitilen ML modelleri, gerçek verinin %87-99 performansında çalışır
2. **Sıfır gerçek kayıt içerir:** Üyelik çıkarım saldırıları AUC=0.51 (rastgele tahmin = bilgi sızıntısı yok)
3. **Karışık veri tiplerini işler:** Hem sayısal hem kategorik sütunlar — hibrit Gaussian+Multinomial difüzyon
4. **Karmaşık veriye ölçeklenir:** 361 boyutlu veri setlerinde test edilmiş

### 4. Teknolojik Temel ve Ön Hazırlık

Bu bir fikir aşamasında değildir — tamamlanmış bir yüksek lisans araştırma projesiyle desteklenmektedir:

**Deneysel Kanıt (Faz 2 — Sistematik Değerlendirme):**
- 10 halka açık veri seti üzerinde 40+ deney (4-361 boyut aralığı)
- 4 yöntem karşılaştırması: Bizim TabDDPM, Vanilya TabDDPM, CTGAN, SMOGN
- 3 değerlendirme boyutu: kullanılabilirlik, istatistiksel sadakat, gizlilik

**Temel Sonuçlar:**
- Yerine koyma senaryosu: Bizim yaklaşımımız ortalama %83-99 kullanılabilirlik
- Genişletme senaryosu: %97-100 kullanılabilirlik (neredeyse gerçek veri performansı)
- Gizlilik: AUC=0.51 (kanıtlanmış güvenlik)
- Ablasyon çalışması: Bizim iyileştirmelerimiz vanilya TabDDPM'ye göre ortalama +4.8 puan

**Kod Tabanı:** ~15.600 satır Python, 58 dosya, PyTorch/scikit-learn tabanlı

**Akademik Referanslar:**
- TabDDPM (Kotelnikov et al., ICML 2023) — temel mimari
- STaSy (Kim et al., ICLR 2023), TabSyn (Zhang et al., ICLR 2024)

### 5. Pazar Fırsatı

**Küresel:** Sentetik veri pazarı 2025'te ~$500M, 2030'da $2.6B+ (CAGR %35). NVIDIA Gretel'ı $320M+'ya satın aldı (Mart 2025), SAS Hazy'yi satın aldı (Kasım 2024) — pazar değeri kanıtlanmış.

**Türkiye:** 
- Sıfır Türk sentetik veri sağlayıcısı (Nisan 2026 doğrulanmış)
- KVKK yaptırımları hızla artıyor: 2024'te tek taramada 503M TL ceza
- Veri gizliliği yönetim pazarı %38.7 CAGR ile büyüyor
- Hiçbir global rakip Türkçe destek veya KVKK'ya özgü özellik sunmuyor

### 6. İş Modeli

**SaaS Platformu:** Aylık abonelik (2.000-5.000 TL/ay professional, 50.000+ TL/yıl enterprise)
**API:** Satır başına kullanım ücreti (platform entegrasyonları için)
**Kurumsal:** Şirket içi dağıtım, özel model eğitimi, uyumluluk raporlama

**Hedef İlk Yıl:** 3-5 pilot müşteri (1 ilaç, 1 finans, 1 üretim), ilk ücretli dönüşümler

### 7. Tahmini Bütçe (1.350.000 TL)

| Kalem | Tutar (TL) | Amaç |
|---|---|---|
| Ürün Geliştirme | 550.000 | Web platformu, API, bulut altyapı |
| GPU/Bulut Altyapı | 200.000 | Model eğitimi, barındırma |
| Ekip | 300.000 | 1 backend + 1 frontend geliştirici (6 ay) |
| Uyumluluk ve Hukuk | 100.000 | A.Ş. kuruluşu, KVKK denetimi, IP tescili |
| Müşteri Keşfi ve Satış | 100.000 | Pilot programlar, sektör etkinlikleri |
| Yedek | 100.000 | Beklenmeyen giderler |
| **Toplam** | **1.350.000** | |

### 8. Girişimci Profili

**Umut Akın**
- Yüksek Lisans — İzmir Yüksek Teknoloji Enstitüsü (yakın zamanda mezun)
- Tez: Difüzyon modelleri ile gizlilik korumalı sentetik tablo verisi üretimi
- Profesyonel: NormDigital — düzenlenmiş sektörler için kurumsal yazılım (ilaç, gümrük)
- Teknik: Python, PyTorch, tam yığın web geliştirme
- Alan kesişimi: Hem teknolojiyi HEM de uyumluluk gereksinimlerini anlar

### 9. Neden Şimdi

- Tablo verisi için difüzyon modelleri son teknoloji (TabDDPM, ICML 2023)
- KVKK yaptırımları Türkiye'de hızlanıyor — 17M TL cezalar, zorunlu DPO atamaları
- Hiçbir Türk rakip pazara girmedi
- Global rakipler konsolide oluyor (satın almalar) — bağımsız, odaklı çözümler için alan açılıyor
- Türk işletmelerde AI/veri okuryazarlığı artıyor

---

## Değerlendirme Kriterlerine Uyum

| Kriter | Puan Hedefi | Güçlü Yönlerimiz |
|---|---|---|
| **İş Fikrinin İçeriği ve Niteliği** | 8-9/10 | Konsept değil — 40+ deneyle doğrulanmış teknoloji |
| **Ürün/Hizmetin Pazar Durumu** | 8-9/10 | $500M+ global pazar, %35 CAGR, Türkiye'de sıfır rakip, KVKK talebi |
| **Proje Ekibinin Yetkinliği ve Uyumu** | 8/10 | Yüksek lisans + NormDigital deneyimi, tam yığın yetkinlik |
| **Ön Hazırlık Çalışmalarının Yeterliliği** | 9-10/10 | Tamamlanmış tez, 15.600 satır kod, 40+ deney, 10 veri seti |

---

## Başvuru Sonrası

- [ ] Taahhütname oluştur (portaldan)
- [ ] Imzala ve DEPARK'a posta ile gönder:
  - DEPARK, Doğuş Caddesi No: 207/Z, DEÜ Tınaztepe Yerleşkesi B, 35000 İzmir
- [ ] Sonuçları bekle → Aşama 2 başlangıcı: 9 Nisan 2026
