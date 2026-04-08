# BiggSinerji İş Fikri Başvuru Formu — Doldurma Kılavuzu

> Bu dosya, Word dokümanına kopyala-yapıştır için hazırlanmıştır.
> Her alan max 3000 karakter.

---

## Proje Yürütücüsü Bilgileri

**Adınız:** Umut
**Soyadınız:** Akın
**T.C. Numaranız:** (kendin doldur)
**Eposta Adresiniz:** umutakin@fastmail.com
**Telefon Numaranız:** 05333491749
**Tebligat Adresi:** Güneşli Mah. 526/5 Sok No: 2 Kat: 3 Daire: 15, Konak/İzmir

**Eğitim Durumu:**
Yüksek Lisans — İzmir Yüksek Teknoloji Enstitüsü (İYTE), Yazılım Mühendisliği ve Veri Bilimleri (SEDS), 2024-2026, GANO: 3.9/4.0
Bilişim Teknolojileri — Ege Üniversitesi, UBE, 2010-2012
Lisans — İzmir Ekonomi Üniversitesi, Bilgisayar Bilimleri / Yazılım Mühendisliği, 2003-2008

**Eş Yatırım Süreci ile İlgileniyor Musunuz?:** Hayır

**BiggSinerji Programını Hangi Kurum Aracılığı ile Duydunuz?:** DEPARK

---

## İş Fikriniz / Girişim Bilgileri

### Girişiminizin Adı

VeriPerde

### İş Fikrinizin Özeti (Max 3000 karakter)

VeriPerde, yapay zeka difüzyon modelleri kullanarak hassas tablo verilerinin gizlilik-güvenli sentetik kopyalarını üreten bir platformdur.

Kuruluşlar hassas verilerini (müşteri kayıtları, üretim parametreleri, finansal veriler) platformumuza yükler. Sistem, orijinal verinin istatistiksel özelliklerini koruyan ancak gerçek kayıt içermeyen sentetik bir veri seti üretir. Bu sentetik veri, KVKK/GDPR riski olmadan serbestçe paylaşılabilir, ML modeli eğitiminde kullanılabilir ve araştırma amaçlı dağıtılabilir.

Teknolojik temel: İYTE'de tamamlanan yüksek lisans araştırmasına dayanmaktadır. TabDDPM mimarisine dayalı hibrit difüzyon modeli (Gaussian + Multinomial) geliştirilmiş, 11 halka açık veri seti üzerinde 44 sistematik deney gerçekleştirilmiştir. 4 farklı yöntem karşılaştırılmıştır (bizim TabDDPM, Vanilla TabDDPM, CTGAN, SMOGN).

Temel sonuçlar:
- Kullanılabilirlik: Sentetik veri üzerinde eğitilen ML modelleri, gerçek verinin %87-99 performansında çalışmaktadır.
- Gizlilik: Üyelik çıkarım saldırılarında AUC=0.51 (rastgele tahmine eşdeğer — bilgi sızıntısı yok).
- Kritik bulgu: Geleneksel yöntemlerin (SMOGN) yüzeysel olarak başarılı görünmelerine rağmen gizlilik sağlamadığı kanıtlanmıştır (AUC>0.80 — saldırganlar eğitim verilerinin %80-92'sini tespit edebilmektedir).

Platform hedefi: Self-servis SaaS platformu olarak sunulacaktır. CSV/Excel yükle → gizlilik/kullanılabilirlik dengesi ayarla → sentetik veri indir. REST API ile mevcut veri akışlarına entegrasyon. Kurumsal müşteriler için şirket içi dağıtım seçeneği.

### Problem Tanımı (Max 3000 karakter)

Sağlık, finans, üretim ve kamu sektöründeki kuruluşlar değerli tablo verisi toplar ancak bu veriyi paylaşamaz:

1. Yasal risk: KVKK, kişisel ve hassas verilerin paylaşımını yasaklar. 2026'da ihlal başına 17,092,242 TL'ye kadar ceza uygulanmaktadır. 2024'te tek bir VERBİS uyum taramasında 503,935,000 TL ceza kesilmiştir (16,350 kuruluş soruşturulmuştur).

2. Ticari sırlar: Üretim parametreleri, fiyatlandırma stratejileri, maliyet yapıları paylaşılamaz.

3. Yeniden kimlik tespiti riski: "Anonimleştirilmiş" veriler tersine mühendislikle çözülebilir. Netflix Prize ve AOL arama kayıtları örneklerinde gösterilmiştir.

Bu durum ciddi iş sonuçları doğurmaktadır:
- İlaç şirketleri klinik araştırma verilerini paylaşamadığı için ilaç geliştirme yavaşlamaktadır.
- Bankalar dolandırıcılık tespit modellerini geliştirmek için işlem verilerini paylaşamamaktadır.
- Üretim şirketleri tedarikçileriyle kalite verilerini paylaşamadığı için optimizasyon fırsatlarını kaçırmaktadır.
- Araştırmacılar gerçekçi verilere erişemedikleri için yapay zeka modellerini geliştiremememektedir.

Mevcut çözümler yetersizdir:
- Anonimleştirme: Tersine çevrilebilir — araştırmalar göstermiştir.
- Veri maskeleme: ML için kullanılabilirliği düşürür.
- GAN tabanlı yaklaşımlar (CTGAN): Deneysel sonuçlarımızda yalnızca %38-93 kullanılabilirlik sağlamıştır ve tutarsızdır.
- SMOGN: Yüksek kullanılabilirlik gösterse de gizlilik sağlamadığı kanıtlanmıştır (AUC>0.80).

Nisan 2025'te KVKK "Yapay Zeka Alanında Kişisel Verilerin Korunması Tavsiyeleri"ni yayımlamıştır. Bu çerçeve, AI sistemlerinin kişisel veri işlemesinde yeni kurallar getirmektedir. Sorun büyümektedir ve acil çözüm gerektirmektedir.

### Çözüm Önerisi (Max 3000 karakter)

VeriPerde, difüzyon modelleri kullanarak hassas tablo verilerinin gizlilik-güvenli sentetik kopyalarını üretir. Difüzyon modelleri — DALL-E gibi görüntü üreticilerin arkasındaki teknoloji — tablo verisine uyarlanmıştır.

Nasıl çalışır:
1. Kullanıcı hassas CSV/Excel dosyasını yükler
2. Sistem verinin istatistiksel yapısını öğrenen bir difüzyon modeli eğitir
3. Model, orijinal veriye istatistiksel olarak eşdeğer ancak gerçek kayıt içermeyen sentetik kayıtlar üretir
4. Kullanıcı gizlilik-güvenli sentetik veri setini indirir
5. Otomatik kalite raporu (kullanılabilirlik, gizlilik, istatistiksel sadakat metrikleri) sunulur

Neden difüzyon modelleri:
- Veri dağılımını öğrenir — interpolasyon yapmaz, gerçekten yeni örnekler üretir
- Hem sayısal hem kategorik sütunları tek bir hibrit modelde işler (Gaussian + Multinomial difüzyon)
- Mod çökmesi riski yoktur (GAN'ların aksine)
- 4-108 boyutlu veri setlerinde test edilmiş ve kanıtlanmıştır

Deneysel kanıt (44 deney, 11 veri seti, 4 yöntem):
- Kullanılabilirlik: %87-99 (sentetik veri üzerinde eğitilen ML modelleri gerçek veriye yakın performans gösterir)
- Gizlilik: AUC=0.51 (üyelik çıkarım saldırısı rastgele tahminden ayırt edilemez)
- Ablasyon: Bizim iyileştirmelerimiz (MinMaxScaler, aykırı değer kırpma, kapasite ölçekleme) vanilya TabDDPM'ye göre ortalama +4.5 puan iyileştirme sağlar

Teknik farklılaşma:
- Rakipler otoregresif modeller (Mostly AI), maskeleme (Tonic.ai) veya LLM tabanlı yaklaşımlar (Gretel/NVIDIA) kullanmaktadır
- Hiçbiri difüzyon modeli kullanmamaktadır
- Difüzyon modelleri tablo verisi için en güncel (state-of-the-art) yaklaşımdır (TabDDPM, ICML 2023)

### Ürün Satış Sürecine Kadarki Planladığınız Aşamalar (Max 3000 karakter)

Aşama 1 — MVP Geliştirme (Ay 1-3):
- Web platformu: Dosya yükleme, model eğitimi, sentetik veri üretimi ve indirme
- REST API: Programatik erişim için dokümante edilmiş API
- Temel kalite raporlama: Kullanılabilirlik, gizlilik ve istatistiksel sadakat metrikleri
- Bulut altyapı: AWS üzerinde GPU hesaplama, güvenli veri depolama

Aşama 2 — Kalite ve Doğrulama (Ay 3-5):
- Otomatik model seçimi ve hiperparametre optimizasyonu
- KVKK uyumluluk raporlama modülü
- Kapsamlı kullanıcı arayüzü iyileştirmeleri
- Beta test programı (3-5 şirket ile)

Aşama 3 — Pilot Programlar (Ay 5-8):
- 3-5 Türk şirket ile ücretli pilot (1 ilaç/sağlık, 1 finans, 1 üretim)
- Vaka çalışmaları ve ROI kanıtları oluşturma
- Müşteri geri bildirimleriyle ürün iyileştirme
- İlk ücretli abonelik dönüşümleri

Aşama 4 — Ölçekleme (Ay 8-12):
- Kurumsal özellikler: Şirket içi dağıtım, SSO, denetim izleri
- Sektör dikey bazında ön-eğitimli modeller (sağlık, finans, üretim)
- Satış ve pazarlama operasyonlarını büyütme
- 10+ ücretli müşteri hedefi

Aşama 5 — Büyüme (Ay 12-18):
- MENA pazarlarına genişleme hazırlığı
- SOC2/ISO 27001 sertifikasyon süreci
- Seri A yatırım turu hazırlığı

### İş Fikrinizin Tematik Alanı

☑ İletişim ve Sayısal Dönüşüm

### Teknoloji Hazırlık Seviyesi (THS/TRL)

☑ 5-6
(Teknoloji laboratuvar ortamında doğrulanmış, gerçekçi ortamda test edilmiş. 44 deney ile kanıtlanmış ancak henüz ticari ürün aşamasında değil.)

### Fikri Mülkiyet (Patent) Durumu

Mevcut patent başvurusu bulunmamaktadır. Proje kapsamında patent başvurusu değerlendirilecektir.

---

## Ürünün/Hizmetin Pazar Durumu

### Müşteri Tanımı (Max 3000 karakter)

Birincil hedef: Düzenlenmiş sektörlerdeki orta ve büyük ölçekli Türk işletmeleri.

Segment 1 — İlaç/Sağlık:
- Klinik araştırma verileriyle çalışan ilaç şirketleri ve CRO'lar
- Hasta verilerini araştırma için paylaşması gereken hastaneler
- Karar vericiler: Veri Koruma Sorumlusu (DPO), IT Direktörü, Klinik Araştırma Müdürü
- Acı noktası: KVKK nedeniyle hasta verileri paylaşılamıyor, ilaç araştırması yavaşlıyor

Segment 2 — Finans/Bankacılık:
- Dolandırıcılık tespit modelleri geliştiren bankalar
- Aktüeryal modelleme için hasar verilerine ihtiyaç duyan sigorta şirketleri
- Karar vericiler: Risk Yönetimi Direktörü, Veri Bilimi Ekip Lideri
- Acı noktası: İşlem verileri ML geliştirme için paylaşılamıyor

Segment 3 — Üretim:
- Tedarikçilerle kalite parametrelerini paylaşması gereken fabrikalar
- Maliyet yapılarını analiz etmek isteyen şirketler
- Karar vericiler: Üretim Müdürü, Kalite Direktörü

Ortak müşteri özellikleri:
- Yıllık 10M+ TL gelir
- KVKK uyumluluk zorunluluğu (DPO atanmış)
- Veri bilimi veya ML ekibi mevcut
- Düzenli veri paylaşım ihtiyacı (iş ortakları, araştırmacılar, iç ekipler arası)

İlk yıl hedefi: 3-5 pilot müşteri ile başlangıç, ilk ücretli dönüşümler.

### Ürün veya Hizmet Tanımı (Max 3000 karakter)

VeriPerde, hassas tablo verilerinin gizlilik-güvenli sentetik kopyalarını üreten bir SaaS platformudur.

Teknik özellikler:
- Hibrit difüzyon modeli: Sayısal sütunlar için Gaussian difüzyon, kategorik sütunlar için Multinomial difüzyon — tek bir birleşik modelde
- Otomatik veri tipi algılama ve ön-işleme
- Yapılandırılabilir gizlilik/kullanılabilirlik dengesi
- 4-108+ boyutlu veri setleri desteği
- CSV, Excel, JSON veri formatları

Fonksiyonel özellikler:
- Self-servis web arayüzü: Veri yükle → yapılandır → sentetik veri indir
- REST API: Programatik erişim, mevcut veri pipeline'larına entegrasyon
- Otomatik kalite raporu: Kullanılabilirlik (R², accuracy), gizlilik (üyelik çıkarım AUC), istatistiksel sadakat (Wasserstein, KS test, korelasyon)
- KVKK uyumluluk raporu: Sentetik verinin kişisel veri içermediğinin kanıtı

Kullanıcıya sağladığı temel fayda:
Kuruluşlar hassas verilerini KVKK riski olmadan paylaşabilir. Sentetik veri üzerinde eğitilen ML modelleri gerçek veriye yakın (%87-99) performans gösterir. Veri paylaşım süreçleri haftalardan dakikalara iner.

Kullanım senaryosu:
Bir ilaç şirketi klinik araştırma verilerini CRO ile paylaşmak istiyor. Gerçek hasta verilerini paylaşamaz (KVKK). VeriPerde'ye veriyi yükler, sentetik versiyonunu üretir. CRO sentetik veri üzerinde ML modeli geliştirir. Model gerçek veride de çalışır (%87-99 performans). Hasta gizliliği korunmuş, araştırma hızlanmış olur.

### Rakipler ve Farklılaşma (Max 3000 karakter)

Küresel rakipler (Nisan 2026):

1. Gretel.ai → NVIDIA tarafından satın alındı (Mart 2025, ~$320M+). LLM tabanlı yaklaşım (Mistral-7B). Artık bağımsız değil, NVIDIA ekosistemine kilitli. Türkiye'de yok.

2. Mostly AI (Avusturya, $31M): Otoregresif model (TabularARGN) kullanıyor. Ocak 2025'te açık kaynak SDK yayımladı. Türkiye'de yok, KVKK desteği yok.

3. Hazy → SAS tarafından satın alındı (Kasım 2024). "SAS Data Maker" oldu. Kurumsal-sadece, SAS Viya üzerinden. SAS'ın Türkiye ofisleri var (dolaylı tehdit).

4. Tonic.ai (ABD, $45M): Çekirdek ürün maskeleme/de-identifikasyon — gerçek sentetik üretim değil. Yeni "Fabricate" ürünü LLM tabanlı.

Türkiye pazarı: Nisan 2026 itibariyle sıfır Türk sentetik veri girişimi doğrulanmıştır (Tracxn, Crunchbase, Startups.watch).

Bizim farklılaşmamız:
- Mimari: Difüzyon modelleri — rakipler otoregresif, GAN veya maskeleme kullanıyor. Difüzyon, tablo verisi için state-of-the-art (ICML 2023).
- Gizlilik kanıtı: AUC=0.51 ile kanıtlanmış gizlilik. Rakiplerin çoğu formal gizlilik testi sunmamaktadır.
- Türkiye'de ilk: KVKK'ya özgü, Türkçe platform.
- Bağımsız: Satın alınan rakiplerin aksine (Gretel→NVIDIA, Hazy→SAS) bağımsız ve odaklı.
- Akademik temel: 44 deney, 11 veri seti ile kanıtlanmış sonuçlar.

### Gelir Modeli (Max 3000 karakter)

SaaS abonelik modeli + API kullanım ücreti + kurumsal lisanslama.

Katman 1 — Starter (Ücretsiz):
- Aylık sınırlı satır (1.000 satır)
- Temel kalite raporu
- Hedef: Araştırmacılar, akademisyenler, değerlendirme amaçlı

Katman 2 — Professional (2.000-5.000 TL/ay):
- Sınırsız veri seti
- Tam kalite ve gizlilik raporlama
- REST API erişimi
- E-posta desteği
- Hedef: KOBİ'ler, veri ekipleri

Katman 3 — Enterprise (Yıllık 50.000+ TL, özel fiyatlandırma):
- Şirket içi dağıtım seçeneği
- Özel model eğitimi ve optimizasyon
- KVKK uyumluluk denetim izleri
- Öncelikli destek ve SLA
- SSO ve kurumsal güvenlik entegrasyonları
- Hedef: Bankalar, ilaç şirketleri, büyük üretim firmaları

Katman 4 — API (Kullanım başına ödeme):
- Satır başına ~0.01-0.10 TL
- Mevcut veri pipeline'larına entegrasyon
- Hedef: Veri platformları, MLOps araçları

Fiyatlandırma stratejisi:
- Global rakipler $295-$3,000+/ay aralığında → Türk pazarı için uygun fiyat noktası
- Freemium model ile düşük müşteri edinme maliyeti
- Yıllık kontrat indirimi (%20)

Gelir projeksiyonu (ilk 3 yıl):
- Yıl 1: ~126.000 TL (3 Professional müşteri)
- Yıl 2: ~780.000 TL (15 Professional + API + 1 Enterprise)
- Yıl 3: ~2.520.000 TL (40 Professional + API + Enterprise)

Başa baş noktası: Yıl 2 sonu — Yıl 3 başı (yıllık ~1.1M TL gelir).

---

## Proje Ekibi Bilgileri

### Ortaklık Yapısı (Max 3000 karakter)

Tek kurucu girişim:
- Umut Akın: %97 (Kurucu, Proje Yürütücüsü, CTO)
- TÜBİTAK BİGG Fonu: %3 (Yatırımcı)

Tüm fikri mülkiyet hakları kurulacak A.Ş.'ye tazminatsız devredilecektir.
Kurucu, fon sonrası tam zamanlı taahhüt verecektir.
YPSS (Yatırım ve Pay Sahipliği Sözleşmesi) şartlarına uyulacaktır.

### Katılmış Olduğunuz Hızlandırma Programları

Daha önce herhangi bir hızlandırma programına, kuluçka merkezine veya hibe/destek programına katılım bulunmamaktadır.

### Ortakların CV'leri

(CV dosyası ek olarak eklenecektir — docs/CV_UMUT_AKIN_TR.md içeriğinden hazırlanan PDF)
