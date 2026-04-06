# AGY112 İş Planı — DRAFT

> **Status:** DRAFT — Not for submission. To be developed during Stage 2 with mentor support.
> **Project:** SynthShield — Privacy-Preserving Synthetic Tabular Data Generation
> **Applicant:** Umut Akin
> **Program:** TÜBİTAK 1812 BİGG / BiggSinerji (DEPARK)

---

## 1. PAZAR FIRSATI (Market Opportunity)

### 1.1 Ürün Tanımı (Product Definition) — 3000 chars

**Problem:** Kuruluşlar (sağlık, finans, üretim, kamu) değerli tablo verisi toplar ancak bu veriyi paylaşamaz. KVKK, GDPR ve HIPAA gibi düzenlemeler kişisel ve hassas verilerin paylaşımını yasaklar. Mevcut çözümler yetersizdir: anonimleştirme tersine çevrilebilir (Netflix Prize, AOL arama kayıtları örnekleri), veri kümeleme bireysel örüntüleri kaybettirir, diferansiyel gizlilik tablo verisi için kullanılabilirliği ciddi ölçüde düşürür.

**Paradoks:** Veri paylaşımı inovasyonu, işbirliğini ve ML gelişimini yönlendirir — ancak ham kayıtları paylaşmak yasal ve etik olarak imkansızdır.

**Çözüm:** SynthShield, yapay zeka difüzyon modelleri kullanarak hassas tablo verilerinin sentetik kopyalarını üreten bir platformdur. Platform:
1. İstatistiksel özellikleri korur — sentetik veri üzerinde eğitilen ML modelleri, gerçek verinin %87-99 performansında çalışır
2. Sıfır gerçek kayıt içerir — üyelik çıkarım saldırıları 0.51 AUC skoru verir (rastgele tahmin = bilgi sızıntısı yok)
3. Karışık veri tiplerini işler — hem sayısal hem kategorik sütunlar
4. Karmaşık veriye ölçeklenir — 361 boyutlu veri setlerinde test edilmiştir

**Kullanıcı Akışı:**
1. Hassas CSV/Excel dosyası yüklenir
2. Gizlilik/kullanılabilirlik dengesi ayarlanır
3. Sistem veriye difüzyon modeli eğitir
4. N adet sentetik kayıt üretilir
5. Gizlilik açısından güvenli sentetik veri seti indirilir
6. KVKK/GDPR riski olmadan serbestçe paylaşılır

**Teknolojik Temel:** Bu bir konsept değildir — tamamlanmış bir yüksek lisans tez projesiyle desteklenir. 10 halka açık veri seti üzerinde 40+ deney, 4 yöntem karşılaştırması, ~15.600 satır kod.

---

### 1.2 Müşteri Tanımı (Customer Definition) — 3000 chars

**Birincil Hedef:** Düzenlenmiş sektörlerdeki orta ve büyük ölçekli Türk işletmeleri:

**Segment 1 — İlaç/Sağlık (Öncelik 1):**
- Klinik araştırma verileriyle çalışan ilaç şirketleri
- Hasta verilerini araştırma için paylaşması gereken hastaneler
- Karar vericiler: Veri Koruma Sorumlusu (DPO), IT Direktörü, Klinik Araştırma Müdürü
- Acı noktası: KVKK uyumluluğu nedeniyle hasta verileri paylaşılamıyor, ilaç araştırması yavaşlıyor

**Segment 2 — Finans/Bankacılık:**
- Dolandırıcılık tespit modelleri geliştiren bankalar
- Aktüeryal modelleme için hasar verilerine ihtiyaç duyan sigorta şirketleri
- Karar vericiler: Risk Yönetimi Direktörü, Veri Bilimi Ekip Lideri
- Acı noktası: İşlem verileri ML geliştirme için paylaşılamıyor

**Segment 3 — Üretim:**
- Tedarikçilerle üretim parametrelerini paylaşması gereken fabrikalar
- Maliyet yapılarını analiz etmek isteyen şirketler
- Karar vericiler: Üretim Müdürü, Kalite Direktörü
- Acı noktası: Ticari sırlar (üretim maliyetleri, fiyatlandırma stratejileri) paylaşılamıyor

**Ortak Özellikler:**
- Yıllık 10M+ TL gelir
- KVKK uyumluluk zorunluluğu (DPO atanmış)
- Veri bilimi veya ML ekibi mevcut
- Düzenli olarak veri paylaşım ihtiyacı var (iş ortakları, araştırmacılar, iç ekipler arası)

---

### 1.3 Müşteri İhtiyaçları (Customer Needs) — 3000 chars

**Temel İhtiyaç:** Yasal risk olmadan hassas verilerin kullanılabilirliğini korumak.

**Mevcut Çözüm Yolları ve Yetersizlikleri:**

| Mevcut Yöntem | Nasıl Kullanılıyor | Neden Yetersiz |
|---|---|---|
| Anonimleştirme | Kimlik bilgileri kaldırılır | Yeniden kimlik tespiti mümkün — araştırmalar göstermiştir |
| Veri maskeleme | Hassas alanlar gizlenir | Bireysel düzeyde örüntüler kaybolur, ML için kullanılamaz |
| Veri kümeleme | Toplu istatistikler paylaşılır | Bireysel kayıt düzeyinde analiz imkansız |
| Manuel erişim kontrolü | Erişim sözleşmelerle sınırlandırılır | Yavaş, bürokratik, paylaşım engellenir |
| Hiçbir şey yapmama | Veri paylaşılmaz | İnovasyon ve işbirliği engellenir |

**Karşılanmayan İhtiyaçlar:**
1. KVKK uyumlu veri paylaşımı — mevcut çözümlerin hiçbiri tam uyumluluk garantisi vermez
2. ML model kalitesinin korunması — anonimleştirme ve maskeleme veri kalitesini düşürür
3. Hız ve otomasyon — manuel süreçler haftalar/aylar sürer
4. Denetlenebilirlik — gizlilik kanıtı ve uyumluluk raporları

**Doğrulama:** Bu ihtiyaçlar NormDigital'deki profesyonel deneyimden (düzenlenmiş sektörler — ilaç, gümrük) ve akademik araştırmadan doğrulanmıştır.

---

### 1.4 Pazar Büyüklüğü (Market Size) — 3000 chars

**Küresel Sentetik Veri Pazarı:**
- 2025 pazar büyüklüğü: ~$450-600M
- 2030 projeksiyonu: $2.6B+ (CAGR ~%35)
- 2034 projeksiyonu: $7.2B+
- Büyüme sürücüleri: GDPR yaptırımı, AI/ML veri ihtiyacı, gizlilik düzenlemelerinin küresel sıkılaştırılması

**Pazar Konsolidasyonu:** 2024-2025'te büyük satın almalar pazar olgunlaşmasını işaret eder:
- NVIDIA, Gretel.ai'yi ~$320M+'ya satın aldı (Mart 2025)
- SAS, Hazy'yi satın aldı (Kasım 2024)

**Türkiye Pazarı:**
- Türkiye veri gizliliği yönetim platformu pazarı: $5.43M (2024), %38.7 CAGR ile büyüyor
- KVKK cezaları hızla artıyor: 2026'da ihlal başına 17,092,242 TL'ye kadar
- 2024'te tek bir VERBİS uyum taramasında 503,935,000 TL ceza kesildi (16,350 kuruluş soruşturuldu)
- Nisan 2025: KVKK "Yapay Zeka Alanında Kişisel Verilerin Korunması Tavsiyeleri" yayımladı

**TAM/SAM/SOM (Türkiye):**
- TAM: ~$50M (Türkiye'deki tüm veri gizliliği harcamaları)
- SAM: ~$10M (sentetik veri çözümlerine ihtiyaç duyan kuruluşlar)
- SOM (İlk Yıl): ~$500K (3-5 pilot müşteri, Starter+Professional katmanları)

**Hedef Sektörler:**
- İlaç/Sağlık, Finans/Bankacılık, Üretim, Kamu, Sigorta

---

### 1.5 Rekabet Durumu (Competitive Landscape) — 3000 chars

**Küresel Oyuncular (Nisan 2026 itibariyle):**

| Şirket | Durum | Yaklaşım | Zayıflık |
|---|---|---|---|
| Gretel.ai → NVIDIA | $320M+ satın alma (Mart 2025) | LLM (Mistral-7B) + GAN + Difüzyon | Artık bağımsız değil; NVIDIA ekosistemine kilitli; Türkiye'de yok |
| Mostly AI | $31M toplam | Otoregresif (TabularARGN) | Açık kaynak SDK yayımladı (Ocak 2025); Türkiye'de yok |
| Hazy → SAS | Satın alma (Kasım 2024) | Gizlilik korumalı üretken modeller | "SAS Data Maker" oldu; kurumsal-sadece; SAS Türkiye ofisleri var (dolaylı tehdit) |
| Tonic.ai | $45M toplam | Maskeleme/de-identifikasyon (çekirdek); LLM tabanlı üretim (yeni) | Çekirdek ürün maskeleme, gerçek sentetik üretim değil |

**Türkiye Pazarı:** Nisan 2026 itibariyle sıfır Türk sentetik veri girişimi doğrulandı (Tracxn, Crunchbase, Startups.watch).

**Rekabet Avantajlarımız:**
1. Difüzyon modelleri — rakipler otoregresif, GAN veya maskeleme kullanıyor
2. Türkiye'de ilk — KVKK'ya özgü, Türkçe platform
3. Bağımsız — satın alınan rakiplerin aksine (Gretel→NVIDIA, Hazy→SAS)
4. Akademik temel — araştırmayla desteklenmiş, kanıtlanmış sonuçlar

---

### 1.6 Rekabet Stratejisi (Competition Strategy) — 3000 chars

**Pazara Giriş Stratejisi: "Türkiye İlk, Sonra Büyüme"**

**Faz 1 — Yerli Pazar Hakimiyeti (Ay 1-12):**
- Türk KOBİ'leri ve orta ölçekli şirketleri hedefle
- KVKK uyumluluğunu birincil satış noktası olarak kullan
- Türkçe destek ve yerelleştirilmiş platform
- 3-5 pilot müşteri ile başla (1 ilaç, 1 finans, 1 üretim)
- Vaka çalışmaları ve referanslar oluştur

**Faz 2 — Kurumsal Büyüme (Ay 12-24):**
- Şirket içi dağıtım seçeneği ekle
- SOC2/ISO 27001 sertifikasyonu al
- Banka ve ilaç şirketleri gibi düzenlenmiş kurumsal müşterileri hedefle
- API marketplace listelemeleri

**Faz 3 — Bölgesel Genişleme (Ay 24+):**
- MENA pazarlarına genişle (benzer düzenleyici ortam)
- AB pazarına GDPR uyumluluğuyla gir
- Sektör dikey bazında ön-eğitimli modeller

**SAS/Hazy Dolaylı Tehdidine Karşı:**
- SAS kurumsal fiyatlandırma ve karmaşıklık — biz KOBİ'lere odaklanıyoruz
- Self-servis platform vs SAS'ın danışmanlık-ağırlıklı yaklaşımı
- Rekabetçi fiyatlandırma (2.000-5.000 TL/ay vs SAS kurumsal kontratları)

---

### 1.7 Engelleyici Faktörler (Blocking Factors) — 3000 chars

| Engel | Ciddiyet | Aşma Planı |
|---|---|---|
| SAS/Hazy mevcut Türkiye ofisleriyle pazara girer | Orta-Yüksek | SAS pahalı ve kurumsal-sadece; biz KOBİ/orta ölçeğe odaklanıyoruz |
| Mostly AI açık kaynak SDK ücretli pazarı baltalar | Orta | Ücretsiz SDK yönetilen servis, KVKK uyumluluğu ve Türkçe destek sunmuyor |
| Müşteriler sentetik veriye güvenmez | Orta | Doğrulama raporları yayımla, kullanılabilirlik garantileri sun, ölçülebilir sonuçlarla pilot programlar |
| Düzenleyici belirsizlik ("sentetik" neyi kapsar?) | Düşük | KVKK/GDPR sentetik veriyi spesifik olarak ele almıyor — gerçekten sentetik veri kişisel veri değildir |
| GPU eğitim maliyeti | Düşük | Bulut GPU fiyatları düşüyor; eğitim veri seti başına tek seferlik |
| Model kalitesi veri setine göre değişir | Orta | Otomatik kalite kontrolleri, teslim öncesi kullanılabilirlik kıyaslamaları |

---

### 1.8 Sosyal Fayda (Social Benefit) — 3000 chars

**BM Sürdürülebilir Kalkınma Hedefleri (SDG) Katkıları:**

**SDG 9 — Sanayi, Yenilik ve Altyapı:**
- Sentetik veri, KOBİ'lerin büyük şirketlerin gizlilik bütçeleri olmadan AI/ML geliştirebilmesini sağlar
- Veri erişimini demokratikleştirir — daha önce gizlilik nedeniyle kilitli olan verileri kullanılabilir kılar

**SDG 3 — Sağlık ve İyi Yaşam:**
- Hasta gizliliğini koruyarak sağlık verilerinin paylaşılmasına olanak tanır
- İlaç araştırması ve klinik deneylerde veri erişimini hızlandırır
- Nadir hastalık araştırmalarında küçük veri setlerinin genişletilmesini sağlar

**SDG 16 — Barış, Adalet ve Güçlü Kurumlar:**
- Kişisel verilerin korunmasını güçlendirir
- Kurumların KVKK uyumluluğuna yardımcı olur
- Veri güvenliğini artırarak vatandaş güvenini destekler

**SDG 8 — İnsana Yakışır İş ve Ekonomik Büyüme:**
- Türkiye'de yeni bir teknoloji sektörü oluşturur (sentetik veri)
- Nitelikli istihdam: veri bilimcileri, ML mühendisleri
- İhracat potansiyeli: MENA ve AB pazarları

**Ek Sosyal Faydalar:**
- Akademik araştırmacılara gerçekçi veri erişimi sağlar (şu an kurumsal veriler erişilemez)
- Startup'ların gizlilik engeli olmadan ML ürünleri geliştirmesini sağlar
- Kamu sektöründe araştırma amaçlı veri paylaşımını mümkün kılar (nüfus, vergi, eğitim verileri)

---

## 2. ÜRÜN VE TEKNOLOJİ (Product & Technology)

### 2.1 Değer Önerisi (Value Proposition) — 3000 chars

**Temel Değer:** Hassas tablo verilerinin gizlilik güvenli sentetik kopyalarını oluştur — gerçek verinin istatistiksel özelliklerini korurken sıfır gerçek kayıt içerir.

**Müşteri İhtiyaçlarını Karşılama:**

1. **KVKK Uyumlu Veri Paylaşımı:**
   - Sentetik veri kişisel veri içermez → KVKK kapsamı dışında
   - Otomatik uyumluluk raporları üretilir
   - Yasal risk sıfıra indirilir

2. **ML Model Kalitesinin Korunması:**
   - Sentetik veri üzerinde eğitilen modeller gerçek verinin %87-99 performansında
   - 10 halka açık veri setinde doğrulanmış
   - Difüzyon modelleri, GAN tabanlı alternatiflerin 2x+ üzerinde performans

3. **Hız ve Otomasyon:**
   - CSV yükle → sentetik veri indir (dakikalar-saatler içinde)
   - Manuel süreçlere (haftalar/aylar) kıyasla dramatik hızlanma
   - REST API ile mevcut veri akışlarına entegrasyon

4. **Doğrulama:**
   - Tamamlanan yüksek lisans tezi — 40+ deney, 10 veri seti, 4 yöntem karşılaştırması
   - Üyelik çıkarım saldırı testi: AUC=0.51 (rastgele tahmin = güvenli)
   - İstatistiksel sadakat metrikleri (Wasserstein, KS test, korelasyon analizi)

---

### 2.2 Teknolojik Rekabet (Technological Competition) — 3000 chars

**Difüzyon Modelleri vs Alternatif Yaklaşımlar:**

| Boyut | GAN Tabanlı (CTGAN) | Otoregresif (Mostly AI) | Maskeleme (Tonic.ai) | Difüzyon (Bizim) |
|---|---|---|---|---|
| Mimari | Üretici-Ayırt Edici eğitimi | Sıralı sütun üretimi | Gerçek veri dönüştürme | İleriye/Geriye difüzyon |
| Kullanılabilirlik | %44-84 | Bilinmiyor (kapalı kaynak) | N/A (maskeleme) | **%87-99** |
| Mod çökmesi riski | Yüksek | Düşük | N/A | **Yok** |
| Karışık veri tipleri | Ayrı pipeline'lar | Doğal destek | Kısıtlı | **Birleşik hibrit difüzyon** |
| Yüksek boyut | Riskli | Bağlam penceresi sınırı | N/A | **361+ boyutta test edildi** |

**Deneysel Kanıt (Faz 2 — 10 Veri Seti):**

Yerine koyma senaryosunda (sentetik ile eğit, gerçek ile test):
- Bizim TabDDPM: Ortalama %83-99 kullanılabilirlik
- Vanilya TabDDPM: Ortalama %78
- CTGAN: Ortalama %77
- SMOGN: Sadece regresyon görevlerinde çalışır; sınıflandırmada basit aşırı örneklemeye düşer

**Ablasyon Çalışması — Bizim İyileştirmelerimiz:**
- MinMaxScaler (QuantileTransformer yerine): +11pp California'da
- Aykırı değer kırpma: Yakınsama kararlılığı iyileşti
- Kapasite ölçekleme: +36.7pp Steel veri setinde
- Ortalama iyileştirme: +4.8pp vanilya TabDDPM üzerinde

---

### 2.3 Ürün Fiyatı (Product Price) — 3000 chars

| Katman | Fiyat | Hedef |
|---|---|---|
| Starter | Ücretsiz (aylık sınırlı satır) | Bireysel araştırmacılar, akademisyenler |
| Professional | 2.000-5.000 TL/ay | KOBİ'ler, veri ekipleri |
| Enterprise | Özel (yıllık 50.000+ TL) | Bankalar, ilaç, kamu |
| API | Çağrı başı (~satır başına 0.01-0.10 TL) | Platform entegrasyonları |

**Fiyatlandırma Varsayımları:**
- Global rakipler $295-$3,000+/ay aralığında fiyatlandırıyor
- Türk KOBİ'ler için uygun fiyat noktası belirlendi
- Freemium model ile müşteri edinme maliyeti düşürülür
- Enterprise katmanında özel model eğitimi ve uyumluluk raporlama dahil

---

### 2.4 Ürün Maliyeti (Product Cost) — 3000 chars

**Birim Maliyet Yapısı (Veri Seti Başına):**

| Maliyet Kalemi | Tutar | Not |
|---|---|---|
| GPU hesaplama (eğitim) | ~5-50 TL | Bulut GPU, veri setinin boyutuna bağlı |
| GPU hesaplama (üretim) | ~0.5-5 TL | Eğitimden çok daha ucuz |
| Depolama | ~0.1-1 TL | Bulut depolama |
| Bant genişliği | ~0.1-0.5 TL | Yükleme/indirme |
| Platform altyapı payı | ~2-10 TL | Sunucu, veritabanı, izleme |
| **Toplam birim maliyet** | **~8-67 TL/veri seti** | |

**Brüt Marj Hesabı:**
- Professional (5.000 TL/ay): ~%90+ brüt marj (aylık ~10-20 veri seti varsayımı)
- Enterprise: Daha yüksek marj (özel model eğitimi ek değer)
- API: ~%80 marj (yüksek hacim, düşük birim maliyet)

---

### 2.5 Tekniğin Bilinen Durumu (State of the Art) — Tablo

| Buluş/Yayın | Referans | Proje Çıktısıyla İlişki | Yenilikçi Yönler |
|---|---|---|---|
| TabDDPM | Kotelnikov et al., ICML 2023 | Temel mimari | Bizim uygulamamız pratik iyileştirmeler ekler: MinMaxScaler, aykırı değer kırpma, kapasite ölçekleme |
| STaSy | Kim et al., ICLR 2023 | Alternatif difüzyon yaklaşımı | Bizim yaklaşımımız daha basit ve uygulanabilir |
| TabSyn | Zhang et al., ICLR 2024 | Gizli uzay difüzyonu | Bizim yaklaşımımız doğrudan veri uzayında çalışır |
| CTGAN | Xu et al., NeurIPS 2019 | Karşılaştırma baseline'ı | Bizim yaklaşımımız 2x+ daha yüksek kullanılabilirlik |

---

### 2.6 Fikri Mülkiyet Hakları (IP Rights) — 3000 chars

**IP Durumu:**
- SEDS500 projesi tamamen Umut Akın'a aittir (kişisel akademik çalışma)
- NormDigital ile IP çakışması yoktur
- Tüm kod, model ve araştırma sonuçları girişimciye aittir

**IP Transfer Planı:**
- Tüm fikri mülkiyet hakları kurulacak A.Ş.'ye devredilecektir
- Devir, 1812 programı gereklilikleri doğrultusunda tazminatsız yapılacaktır
- Kod tabanı, eğitilmiş modeller, araştırma sonuçları ve marka hakları dahil

**Koruma Stratejisi:**
- Patent başvurusu: Difüzyon tabanlı tablo verisi üretim yöntemi (değerlendirilecek)
- Marka tescili: "SynthShield" adı ve logosu
- Ticari sır koruması: Eğitim optimizasyonları, ön-işleme yenilikleri (know-how)
- Açık kaynak stratejisi: Çekirdek algoritma açık kaynak değil; API erişimi sağlanır

---

### 2.7 Regülasyonlar (Regulations) — 3000 chars

**İlgili Düzenlemeler:**

**KVKK (6698 Sayılı Kanun):**
- Kişisel verilerin işlenmesi, saklanması ve paylaşılması düzenler
- Sentetik veri, gerçek kişilerle ilişkilendirilemediği sürece KVKK kapsamı dışındadır
- Ancak: Giriş verisi (orijinal hassas veri) KVKK kapsamındadır — platform güvenliği kritik
- KVKK uyumluluk planı: Veri minimizasyonu, erişim kontrolü, işlem kayıtları, DPO danışmanlığı

**GDPR (AB pazarı için):**
- Recital 26: Anonim veriler GDPR kapsamı dışında
- Sentetik verinin "yeterince anonim" olması gerekir — üyelik çıkarım testi ile doğrulanır
- AB pazarına giriş için GDPR uyumluluk değerlendirmesi yapılacak

**Sektörel Düzenlemeler:**
- Sağlık: Sağlık verilerinin paylaşımında KVKK madde 6 (özel nitelikli veriler)
- Finans: BDDK düzenlemeleri, banka sırrı
- Her sektör için uyumluluk şablonları hazırlanacak

**Nisan 2025 Gelişmesi:**
- KVKK, "Yapay Zeka Alanında Kişisel Verilerin Korunması Tavsiyeleri" yayımladı
- AI sistemlerinin kişisel veri işlemesinde çerçeve belirledi
- Platformumuz bu tavsiyelere uyumlu tasarlanacak

---

## 3. EKİP (Team)

### 3.1 Ekip ve Deneyim (Team & Experience) — 3000 chars

**Kurucu — Umut Akın:**
- Yüksek Lisans: İzmir Yüksek Teknoloji Enstitüsü (İYTE) — yakın zamanda mezun
- Tez konusu: Difüzyon modelleri ile gizlilik korumalı sentetik tablo verisi üretimi
- Profesyonel deneyim: NormDigital — düzenlenmiş sektörler için kurumsal yazılım (ilaç, gümrük)
- Teknik yetkinlikler: Python, PyTorch, scikit-learn, tam yığın web geliştirme
- Alan uzmanlığı: Hem teknoloji HEM de uyumluluk gereksinimlerini anlar

**Akademik Danışman — Dr. Damla Oğuz (İYTE):**
- Araştırma danışmanı olarak destek sağlayacak
- Akademik ağ ve referans

**Teknik Ekip (Fon Sonrası):**
- 1 Backend Geliştirici (6 ay): API ve altyapı
- 1 Frontend Geliştirici (6 ay): Web platformu

**Neden Bu Ekip:**
- Kurucu tüm ürünü tek başına geliştirebilir (ML + web) — ekip büyümesi isteğe bağlı
- NormDigital deneyimi düzenlenmiş sektör müşterilerini anlamayı sağlar
- Akademik temel araştırma derinliği ve güvenilirlik sağlar

---

### 3.2 Ortaklık Yapısı (Partnership Structure) — 3000 chars

**Planlanan A.Ş. Ortaklık Yapısı:**

| Ortak | Pay (%) | Rol |
|---|---|---|
| Umut Akın | %97 | Kurucu, CEO, Baş Teknoloji Sorumlusu |
| TÜBİTAK BİGG Fonu | %3 | Yatırımcı |

**Taahhütler:**
- Tüm IP'nin tazminatsız şirkete devri
- Fon sonrası tam zamanlı taahhüt
- YPSS (Yatırım ve Pay Sahipliği Sözleşmesi) şartlarına uyum

---

### 3.3 İş Paylaşımı (Work Distribution) — 3000 chars

**Kurucu — Umut Akın (Tam Zamanlı, 18 ay):**
- Ürün geliştirme ve teknik liderlik
- ML model geliştirme ve optimizasyon
- Müşteri keşfi ve pilot programlar
- İş geliştirme ve satış
- KVKK uyumluluk çalışmaları

**Backend Geliştirici (6 ay kontrat):**
- REST API geliştirme
- Veritabanı ve iş kuyruğu altyapısı
- Bulut dağıtımı ve DevOps

**Frontend Geliştirici (6 ay kontrat):**
- Web arayüzü (upload, konfigürasyon, indirme)
- Dashboard ve raporlama arayüzü
- Kullanıcı yönetimi

---

### 3.4 İşbirlikleri (Collaborations) — 3000 chars

**Akademik İşbirliği:**
- Dr. Damla Oğuz (İYTE) — araştırma danışmanlığı, akademik doğrulama
- İYTE Bilgisayar Mühendisliği Bölümü — stajyer ve araştırma asistanı potansiyeli

**Sektör İşbirliği:**
- DEPARK (Dokuz Eylül Teknoloji Geliştirme Bölgesi) — mentorluk ve ağ
- Potansiyel pilot müşteriler — ilaç, finans ve üretim sektöründen 3-5 şirket

**Teknoloji İşbirliği:**
- Bulut hizmet sağlayıcıları (AWS/GCP) — GPU hesaplama altyapısı
- Açık kaynak topluluk — PyTorch, scikit-learn ekosistemi

---

### 3.5 Ekibin Misyonu ve Değerleri (Team Mission & Values) — 3000 chars

**Misyon:** Kuruluşların veri gizliliğini korurken verilerinin tam potansiyelini kullanmasını sağlamak. Hassas verilerin paylaşım engellerini kaldırarak inovasyonu hızlandırmak.

**Değerler:**
1. **Gizlilik Öncelikli:** Her kararımızda veri gizliliği en üst önceliktir
2. **Bilimsel Titizlik:** İddialarımızı deneysel kanıtlarla destekleriz
3. **Şeffaflık:** Sonuçlarımızı — başarısızlıklar dahil — açıkça paylaşırız
4. **Müşteri Odaklılık:** Gerçek müşteri sorunlarını çözeriz, teknoloji gösterişi yapmayız
5. **Sürekli Öğrenme:** Alandaki en son gelişmeleri takip eder ve uygularız

---

## 4. İŞ MODELİ VE FİNANSAL ÖNGÖRÜLER

### 4.1 İş Modeli (Business Model) — 3000 chars

**Gelir Kaynakları:**

**Katman 1 — SaaS Platformu (Birincil):**
- Self-servis web platformu: veri yükle, sentetik versiyon üret, indir
- Aylık abonelik modeli
- Hedef: KOBİ'ler, startup'lar, araştırma laboratuvarları

**Katman 2 — Kurumsal (Büyüme):**
- Şirket içi dağıtım (düzenlenmiş sektörler için)
- Özel model eğitimi ve doğrulama
- Uyumluluk raporlama (KVKK/GDPR denetim izi)
- Hedef: Bankalar, ilaç şirketleri, hastaneler

**Katman 3 — API (Ölçekleme):**
- REST API ile mevcut veri pipeline'larına entegrasyon
- Kullanım başına ödeme fiyatlandırması
- Hedef: Veri platformları, MLOps araçları, analitik şirketleri

---

### 4.2 Müşteriye Erişim (Customer Acquisition) — 3000 chars

**Kanal Stratejisi:**

1. **İçerik Pazarlama ve Düşünce Liderliği:**
   - Blog yazıları: KVKK uyumluluğu, sentetik veri kullanım örnekleri
   - Teknik makaleler ve vaka çalışmaları
   - Webinar'lar ve konferans sunumları

2. **Doğrudan Satış (Enterprise):**
   - DEPARK ağı üzerinden tanıtım
   - Sektör etkinlikleri ve fuarlar
   - Referans programı

3. **Freemium Dönüşüm:**
   - Ücretsiz katman ile müşteri edinme
   - Kullanım arttıkça ücretli katmana geçiş
   - Self-servis onboarding

4. **Pilot Programları:**
   - 3-5 şirketle ücretsiz pilot (3 ay)
   - Ölçülebilir sonuçlar ve ROI kanıtı
   - Başarılı pilotları ödeme yapan müşterilere dönüştür

---

### 4.3 Kritik İş Adımları (Critical Business Steps) — Tablo

| Görev | Tamamlanma Tarihi | Doğrulanabilir Başarı Kriteri |
|---|---|---|
| MVP web platformu tamamlanması | 2026-09 | Çalışan upload/generate/download akışı |
| REST API yayına alma | 2026-09 | Dokümante edilmiş API, kimlik doğrulama |
| İlk pilot müşteri başlatma | 2026-10 | 1 şirket sentetik veri üretimi yapıyor |
| 3 pilot müşteri tamamlanma | 2026-12 | 3 vaka çalışması, kullanılabilirlik raporları |
| Ücretli müşteri dönüşümü | 2027-01 | İlk ücretli abonelik |
| 10 ücretli müşteri | 2027-06 | Aylık tekrarlayan gelir (MRR) başarısı |
| Kurumsal özellikler (on-prem, SSO) | 2027-09 | 1 kurumsal dağıtım |
| MENA pazarına giriş | 2028-06 | 1 yabancı müşteri |
| Seri A yatırım turu | 2028-12 | Yatırım kapanışı |

---

### 4.4 Başa Baş Noktası Analizi (Break-Even Analysis)

**Tablo 4.4-1: Gelirler (Yıllık Projeksiyon)**

| | Yıl 1 | Yıl 2 | Yıl 3 | Yıl 4 | Yıl 5 |
|---|---|---|---|---|---|
| Professional müşteri sayısı | 3 | 15 | 40 | 80 | 150 |
| Ortalama aylık gelir/müşteri (TL) | 3.500 | 3.500 | 4.000 | 4.000 | 4.500 |
| API geliri (TL) | 0 | 50.000 | 200.000 | 500.000 | 1.000.000 |
| Enterprise geliri (TL) | 0 | 100.000 | 400.000 | 1.000.000 | 2.000.000 |
| **Toplam Gelir (TL)** | **126.000** | **780.000** | **2.520.000** | **4.340.000** | **11.100.000** |

**Tablo 4.4-2: Değişken Maliyetler**

| | Birim |
|---|---|
| GPU hesaplama | ~10-50 TL/veri seti |
| Bulut altyapı | ~500 TL/müşteri/ay |

**Tablo 4.4-3: Sabit Maliyetler (Yıllık)**

| | Tutar (TL) |
|---|---|
| Kurucu maaş | 300.000 |
| Geliştirici (1-2 kişi) | 400.000 |
| Ofis/DEPARK | 60.000 |
| Bulut altyapı (temel) | 120.000 |
| Hukuk/Muhasebe | 50.000 |
| **Toplam Sabit** | **930.000** |

**Tablo 4.4-4: Başa Baş Analizi**
- Kar marjı: ~%85 (brüt)
- Başa baş noktası: ~1.100.000 TL gelir / yıl
- Beklenen başa baş zamanı: **Yıl 2 sonu — Yıl 3 başı**

---

### 4.5 Yatırımcı İlişkileri (Investor Relations) — 3000 chars

**Mevcut Yatırım:** TÜBİTAK BİGG Fonu — 1.350.000 TL karşılığında %3 pay

**Sonraki Turlar:**
- Pre-seed (Yıl 1 sonu): 500K-1M TL — melek yatırımcılar, DEPARK ağı
- Seed (Yıl 2): 3-5M TL — Türk VC fonları (örn: 500 Istanbul, Revo Capital)
- Seri A (Yıl 3-4): 15-30M TL — bölgesel/global VC

**Çıkış Stratejisi:**
- Birincil: Stratejik satın alma (veri platformu şirketleri, bulut sağlayıcıları)
- İkincil: Bölgesel lider olarak büyüme ve temettü
- Karşılaştırma: NVIDIA Gretel'ı $320M+'ya, SAS Hazy'yi satın aldı — pazar M&A aktivitesi yüksek

---

## 5. RİSK YÖNETİMİ (Risk Management)

### 5.1 Riskler ve Önlemler — Tablo

| Risk | Olasılık | Etki | Önlemler | B Planı |
|---|---|---|---|---|
| SAS/Hazy Türkiye pazarına girer | Orta | Yüksek | KOBİ/orta ölçeğe odaklan, rekabetçi fiyatlandırma, KVKK uzmanlığı | Niş sektörlere (küçük ilaç, yerli üretim) odaklan |
| Mostly AI açık kaynak SDK pazar payı alır | Orta | Orta | Yönetilen servis değeri, KVKK uyumluluğu, Türkçe destek | SDK üzerine katma değerli servisler sun |
| Müşteriler sentetik veriye güvenmez | Orta | Yüksek | Doğrulama raporları, pilot programlar, akademik referanslar | Mevcut müşteri referansları ve vaka çalışmaları |
| GPU maliyetleri beklentinin üstünde | Düşük | Düşük | Bulut fiyat optimizasyonu, spot instance kullanımı | Model sıkıştırma ve optimizasyon |
| Kilit personel kaybı | Düşük | Yüksek | Rekabetçi ücret, hisse opsiyonu, misyon odaklı kültür | Bilgi belgeleme, kod dokümantasyonu |
| Düzenleyici değişiklik (sentetik veri KVKK kapsamına alınır) | Düşük | Orta | Düzenleyici gelişmeleri takip et, proaktif uyumluluk | Platform güvenlik seviyesini artır, şifreleme |
