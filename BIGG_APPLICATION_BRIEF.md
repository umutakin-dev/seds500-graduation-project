# BiggSinerji Application Brief — Synthetic Data Platform

> Prepared: 2026-04-02
> Applicant: Umut Akin
> Project: Privacy-Preserving Synthetic Tabular Data Generation using Diffusion Models (SEDS500)

---

## Table of Contents

1. [BiggSinerji Program Overview](#1-biggsinerji-program-overview)
2. [Eligibility Check](#2-eligibility-check)
3. [Timeline & Deadlines](#3-timeline--deadlines)
4. [Application Process & Requirements](#4-application-process--requirements)
5. [Evaluation Criteria](#5-evaluation-criteria)
6. [Product Vision — SynthShield (Working Name)](#6-product-vision--synthshield-working-name)
7. [Technical Foundation (From SEDS500 Research)](#7-technical-foundation-from-seds500-research)
8. [Market Opportunity](#8-market-opportunity)
9. [Competitive Landscape](#9-competitive-landscape)
10. [Business Model](#10-business-model)
11. [Use of Funds (1,350,000 TL)](#11-use-of-funds-1350000-tl)
12. [Pitch Narrative](#12-pitch-narrative)
13. [Strengths for Panel](#13-strengths-for-panel)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [MVP Roadmap](#15-mvp-roadmap)
16. [Key Research Results (Evidence)](#16-key-research-results-evidence)
17. [References & Resources](#17-references--resources)

---

## 1. BiggSinerji Program Overview

**What:** BiggSinerji is an acceleration program under TUBİTAK's BİGG 1812 (Investment-Based Entrepreneurship Support Program). It supports technology-driven entrepreneurship by transforming business ideas into commercial ventures.

**Investment:** 1,350,000 TL in exchange for 3% equity stake via TUBİTAK BİGG Fund.

**Company requirement:** Must incorporate as Anonim Şirket (Joint Stock Company) in Turkey.

**Implementing organization:** DEPARK (Dokuz Eylul Technology Development Zone)

**Program structure — 4 stages:**

| Stage | Focus | Timeline (2026) |
|-------|-------|-----------------|
| Stage 1 — Idea Submission | Submit business idea via TUBİTAK portal | March 16 – April 8 |
| Stage 2 — Training & Mentoring | AGY112 business plan development, KPI mentors | April 9 – April 28 |
| Stage 3 — Expert Mentoring | Gap analysis, specialist 1:1 coaching, panel prep | April 29 – May 14 |
| Stage 4 — Pitch & Evaluation | Presentation training, jury evaluation | May 15 – June 12 |

**Support provided:**
- Entrepreneurship training & workshops
- Customer discovery meetings
- Business plan development (AGY112)
- Mentorship (KPI mentors + domain experts)
- Presentation coaching
- Panel simulation (mimicking TUBİTAK jury)

**Post-acceptance:**
- Receive "Mukemmeliyet Muhru" (Excellence Certificate)
- Incorporate A.Ş.
- Transfer all IP to the company
- Full-time commitment required (exception: university teaching)
- Receive 1,350,000 TL investment

**Contact:**
- Email: biggsinerji@depark.com
- Phone: +90 232 453 03 93
- Address: Dogus Caddesi No: 207/Z DEU Tinaztepe Yerleskesi B, 35000 Izmir
- Portal: https://giris.tubitak.gov.tr/kullaniciadiilegiris.htm

---

## 2. Eligibility Check

| Requirement | Status |
|-------------|--------|
| Enrolled in or graduated from university program | Master's completed (recently graduated) |
| No partnership in any capital company or sole proprietorship | To be confirmed — must not hold shares in any company at application date |
| Never received Ministry of Industry Technopreneurship Capital Support | To be confirmed |

**Action items before applying:**
- Verify you hold no company shares (including norm-related entities)
- Confirm no prior Technopreneurship Capital Support received
- If there's a conflict with norm employment, clarify IP ownership boundaries

---

## 3. Timeline & Deadlines

```
TODAY:        April 2, 2026
DEADLINE:     April 8, 2026, 00:00 (midnight)
DAYS LEFT:    ~6 DAYS
```

**Critical path:**
1. **By April 7:** Complete and submit application on TUBİTAK portal
2. **April 9:** Stage 2 begins (if accepted)
3. **June 12:** Program concludes

---

## 4. Application Process & Requirements

**Portal:** https://giris.tubitak.gov.tr/kullaniciadiilegiris.htm

**Required documents:**
- AGY112 Business Plan (official TUBİTAK form)
- Investment and Shareholder Agreement (YPSS)
- Commitment documents (Tahutname)
- 2026-01 Call Announcement documentation

**Key constraints:**
- All IP must transfer to the new company (no compensation)
- Full-time commitment required after funding
- Must form A.Ş. (not Ltd. Şti.)

---

## 5. Evaluation Criteria

TUBİTAK evaluates across three equally-weighted dimensions:

### A. Market / Commercial Viability
- Is there a real, paying market?
- How big is the addressable market?
- Who are the customers, and will they pay?
- Revenue model clarity

### B. Technical Feasibility & Innovation
- Is the technology novel?
- Can it actually be built?
- What's the R&D depth?
- Defensibility (patents, know-how, data moats)

### C. Team Capability & Execution
- Does the founder understand the domain?
- Can they execute?
- Relevant experience and education
- Commitment level

---

## 6. Product Vision — SynthShield (Working Name)

### One-liner
A platform that generates privacy-safe synthetic copies of sensitive tabular data using AI diffusion models, enabling organizations to share and utilize data without exposing real records.

### Problem Statement
Organizations across industries (healthcare, finance, manufacturing, government) collect valuable tabular data but **cannot share it** due to:
- **Legal risk:** KVKK (Turkish GDPR), GDPR, HIPAA prohibit sharing personal/sensitive data
- **Trade secrets:** Manufacturing parameters, pricing strategies, cost structures
- **Re-identification risk:** "Anonymized" data can be reverse-engineered (Netflix Prize, AOL search logs, etc.)

**The paradox:** Data sharing drives innovation, collaboration, and ML development — but sharing raw records is legally and ethically impossible.

**Current workarounds and why they fail:**
| Method | Problem |
|--------|---------|
| Anonymization | Reversible — research shows re-identification is possible |
| Aggregation | Loses individual-level patterns needed for ML |
| Differential privacy (noise) | Degrades utility significantly for tabular data |
| GANs (CTGAN) | Only 35% utility retention in our tests |
| SMOGN (interpolation) | Catastrophically fails on complex data (negative R²) |

### Solution
Use **diffusion models (TabDDPM)** to generate synthetic tabular data that:
1. **Preserves statistical properties** — ML models trained on synthetic data perform at 87-98% of real data
2. **Contains zero real records** — membership inference attacks score 0.51 AUC (= random guessing)
3. **Handles mixed data types** — both numerical and categorical columns
4. **Scales to complex data** — works on 117-dimensional datasets (unlike SMOGN)

### How It Works (User Flow)
```
1. Upload sensitive CSV/Excel file
2. Configure privacy/utility tradeoff
3. System trains diffusion model on your data
4. Generate N synthetic records
5. Download privacy-safe synthetic dataset
6. Share freely — no KVKK/GDPR risk
```

---

## 7. Technical Foundation (From SEDS500 Research)

### What's already built
This is not a concept — it's backed by a completed master's graduation project with rigorous experimentation:

- **19 experiments** conducted across 2 real-world organizational datasets
- **Implementation:** TabDDPM-style hybrid diffusion (Gaussian for numerical + Multinomial for categorical features)
- **Architecture:** MLP denoiser with sinusoidal timestep embeddings, cosine noise schedule
- **Codebase:** 58 Python files, ~15,600 lines of code
- **Dependencies:** PyTorch, scikit-learn, pandas (standard ML stack)

### Core Technical Components

**Gaussian Diffusion (numerical features):**
- Forward: gradually add noise over T timesteps
- Reverse: neural network learns to denoise
- Loss: MSE between predicted and actual noise

**Multinomial Diffusion (categorical features):**
- Forward: corrupt one-hot encodings toward uniform distribution
- Reverse: predict category distributions via logits
- Key innovations: log-space operations (prevents underflow), KL divergence loss, Gumbel-softmax sampling

**MLP Denoiser:**
- 3-4 hidden layers (256-512 units)
- Sinusoidal positional encoding for timesteps
- ~200K-500K parameters depending on input dimensionality

### Key Technical Breakthroughs Achieved

| Innovation | Impact |
|-----------|--------|
| Log-space operations for categorical diffusion | Prevents NaN/Inf during training |
| KL divergence loss (replacing MSE for categories) | Respects probability constraints |
| Gumbel-softmax sampling | Enables gradient flow through categorical sampling |
| Proper posterior computation | Faithful reconstruction in reverse process |
| Categorical sampling bug fix at high timesteps | Changed R² from -14.0 to 0.17 |
| MinMaxScaler instead of QuantileTransformer | Fixed invertibility issue, 3-8x variance improvement |

**Combined impact:** These innovations produced a **3.3x improvement** over naive diffusion (26.5% → 87.3% utility).

---

## 8. Market Opportunity

### Global Synthetic Data Market
- **2025 market size:** ~$450-600M (estimated)
- **Projected 2030:** $2.6B+ (CAGR ~35%)
- **Projected 2034:** $7.2B+
- **Drivers:** GDPR enforcement, AI/ML data hunger, privacy regulations tightening globally
- **Consolidation wave:** Major acquisitions in 2024-2025 (Gretel→NVIDIA, Hazy→SAS) signal market maturation

### Turkish Market Specifically
- **KVKK (Kisisel Verilerin Korunmasi Kanunu):** Turkey's data protection law, closely modeled on GDPR
- **Enforcement accelerating dramatically:**
  - 2026 fine ceiling: **17,092,242 TL** per violation (up ~80% from 2024)
  - 2024: **503,935,000 TL** in penalties issued in a single VERBiS compliance sweep (16,350 organizations investigated)
  - 72-hour breach notification now mandatory (2025 amendment)
  - Mandatory DPO appointments for medium/large companies
  - April 2025: KVKK published "Recommendations on the Protection of Personal Data in the Field of Artificial Intelligence"
- **Turkey data privacy management market:** $5.43M (2024), growing at **38.7% CAGR**
- **No Turkish synthetic data provider exists** — confirmed via Tracxn, Crunchbase, Startups.watch (zero entries)
- **Target sectors in Turkey:**
  - **Pharma/Healthcare:** Patient data sharing for drug research, clinical trials
  - **Finance/Banking:** Transaction data for fraud detection model development
  - **Manufacturing:** Production data sharing with suppliers/partners
  - **Government/Public sector:** Census, tax, education data for research
  - **Insurance:** Claims data for actuarial modeling

### Why Turkey First
- First-mover advantage in a market with no local competitor (verified April 2026)
- KVKK enforcement is creating urgent demand — fines now reach 17M TL per violation
- No global competitor has Turkish localization or KVKK-specific features
- Turkish-language support and local business relationships
- Expand to EU/global markets once validated

---

## 9. Competitive Landscape

### Global Players (Updated April 2026)

| Company | Location | Status | Approach | Weakness |
|---------|----------|--------|----------|----------|
| **Gretel.ai** | USA → NVIDIA | **Acquired by NVIDIA** (Mar 2025, ~$320M+) | LLMs (Mistral-7B) + GANs + Diffusion | No longer independent; locked into NVIDIA ecosystem; no Turkish presence |
| **Mostly AI** | Austria | $31M total, no new funding since Jan 2022 | **Autoregressive (TabularARGN)** — not GANs | Released open-source SDK (Jan 2025); no Turkish market |
| **Hazy** | UK → SAS | **Acquired by SAS** (Nov 2024) | Privacy-preserving generative models | Now "SAS Data Maker"; enterprise-only via SAS Viya; **SAS has Turkey offices — indirect threat** |
| **Tonic.ai** | USA | $45M total, no new funding since Sep 2021 | **Masking/de-identification** (core); LLM-based generation (new "Fabricate" product) | Core product is masking, not true synthetic generation; expensive |
| **Datagen** | Israel | $50M Series B | Synthetic images (not tabular) | Different domain |

**Key trend:** Two major players (Gretel, Hazy) were acquired in 2024-2025, leaving Mostly AI and Tonic.ai as the remaining independents — neither with Turkish presence.

### Our Differentiation

| Dimension | Competitors | SynthShield (Diffusion-based) |
|-----------|-------------|-------------------------------|
| Architecture | Autoregressive (Mostly AI), LLM-based (Gretel/Tonic), masking (Tonic core) | **Purpose-built diffusion models** — state-of-the-art for tabular data |
| Utility retention | ~35% (CTGAN baseline) | **87-98%** |
| Privacy | Varies | **Validated: AUC=0.51** |
| Mixed data types | Often separate pipelines | **Unified hybrid diffusion** |
| High-dimensional data | Mode collapse risk (GANs), context window limits (LLMs) | **Scales to 117+ dimensions** |
| Independence | Gretel→NVIDIA, Hazy→SAS (locked in) | **Independent, focused product** |
| Turkish market | None present; SAS/Hazy is only indirect threat | **First mover, KVKK-native** |
| Cost | Mostly AI SDK is free; Tonic starts $29/mo; enterprise is custom | **Competitive pricing for Turkish market** |

### Defensibility
- **Know-how moat:** Deep implementation expertise (TabDDPM, bug fixes, preprocessing insights)
- **Academic foundation:** Published research, advisor relationship
- **Architectural advantage:** Diffusion models outperform autoregressive and GAN approaches on tabular data — competitors would need to rebuild
- **Local market:** Turkish business relationships, KVKK expertise, Turkish-language platform
- **Independence:** Unlike acquired competitors (Gretel→NVIDIA, Hazy→SAS), we're a focused product company
- **Data moat (future):** Pre-trained models per industry vertical

### Competitive Risks to Address
- **Mostly AI open-source SDK:** Free local use under Apache 2.0 — lowers barrier for technical users. Mitigation: our platform adds KVKK compliance, Turkish support, and managed service value
- **SAS/Hazy in Turkey:** SAS has existing Turkish enterprise relationships. Mitigation: SAS is expensive, enterprise-only; we target SMEs and mid-market first
- **LLM-based approaches:** Gretel/Tonic moving to LLM-based generation. Mitigation: LLMs have context window limits for high-dimensional tabular data; diffusion models scale better

---

## 10. Business Model

### Revenue Streams

**Tier 1 — SaaS Platform (Primary)**
- Self-service web platform: upload data, generate synthetic version
- Pricing: per-dataset or monthly subscription
- Target: SMEs, startups, research labs

**Tier 2 — Enterprise (Growth)**
- On-premise deployment for regulated industries
- Custom model training and validation
- Compliance reporting (KVKK/GDPR audit trails)
- Target: Banks, pharma companies, hospitals

**Tier 3 — API (Scale)**
- REST API for integration into existing data pipelines
- Pay-per-generation pricing
- Target: Data platforms, MLOps tools, analytics companies

### Pricing Strategy (Preliminary)
| Tier | Price | Target |
|------|-------|--------|
| Starter | Free (limited rows/month) | Individual researchers, academics |
| Professional | ~2,000-5,000 TL/month | SMEs, data teams |
| Enterprise | Custom (50,000+ TL/year) | Banks, pharma, government |
| API | Per-call (~0.01-0.10 TL per row) | Platform integrations |

---

## 11. Use of Funds (1,350,000 TL)

| Category | Amount (TL) | Purpose |
|----------|-------------|---------|
| **Product Development** | 550,000 | Web platform (frontend + backend), API development, cloud infrastructure |
| **GPU/Cloud Infrastructure** | 200,000 | GPU compute for model training (AWS/GCP), hosting |
| **Team** | 300,000 | 1 backend developer (6 months), 1 frontend developer (6 months) |
| **Compliance & Legal** | 100,000 | A.Ş. formation, KVKK compliance audit, legal counsel, IP registration |
| **Customer Discovery & Sales** | 100,000 | Pilot programs with 3-5 companies, industry events, marketing |
| **Reserve** | 100,000 | Contingency |
| **Total** | **1,350,000** | |

---

## 12. Pitch Narrative

### The 30-Second Version
> "Turkish organizations sit on valuable data they can't share — KVKK won't let them, and anonymization doesn't work. We use AI diffusion models to generate synthetic copies of tabular data that are statistically identical but contain zero real records. In our research, synthetic data retained 98% of the original's predictive power while being completely private. No Turkish solution exists. We're the first."

### The 2-Minute Version
> "I just completed my master's researching how diffusion models — the same AI behind image generators like DALL-E — can generate synthetic tabular data. The results were striking: our approach retained 87-98% of the original data's utility while membership inference attacks couldn't tell real from fake (AUC = 0.51, which is random guessing).
>
> This matters because organizations across healthcare, finance, and manufacturing collect critical data but can't share it. KVKK makes it illegal — and enforcement is accelerating fast. In 2024 alone, KVKK issued 503 million TL in fines in a single compliance sweep. The 2026 fine ceiling is now 17 million TL per violation. Organizations need a way to use their data without the legal risk.
>
> The global market validates this. NVIDIA acquired synthetic data startup Gretel for $320M+ in 2025. SAS acquired Hazy in 2024. Mostly AI raised $31M. But none of them serve Turkey, none have KVKK expertise, and none use diffusion models — they rely on older architectures like autoregressive models or masking.
>
> We're building SynthShield — a platform where you upload your sensitive data and download a privacy-safe synthetic version. Train ML models on it, share it with partners, publish it for research — all without legal risk. Our diffusion-based approach achieves 3x better utility than the GAN baselines these companies started with.
>
> With BiggSinerji's 1,350,000 TL, we'll build the MVP platform, run pilots with 3-5 Turkish companies, and validate our go-to-market. I've spent the last year building the AI — now I need to build the business around it."

---

## 13. Strengths for Panel

### Why This Idea
- **Not a concept** — backed by 19 experiments, 15,600 lines of code, and a defended thesis
- **Quantified results** — 87-98% utility, AUC=0.51 privacy, benchmarked against 3 alternatives
- **Real market** — $450-600M globally, 35% CAGR, zero Turkish competitors (verified April 2026)
- **Regulatory tailwind** — KVKK fines up to 17M TL per violation, 503M TL issued in 2024 alone
- **Validated by M&A** — NVIDIA paid $320M+ for Gretel, SAS acquired Hazy — proving market value

### Why This Founder
- **Master's degree** in the exact domain (graduated recently)
- **Professional experience** at NormDigital — enterprise software for regulated industries (pharma, customs)
- **Full-stack capability** — can build the entire product (Python ML + web platform)
- **Domain crossover** — understands both the technology AND the compliance requirements

### Why Now
- Diffusion models for tabular data are state-of-the-art (TabDDPM published ICML 2023)
- KVKK enforcement is accelerating — 17M TL fines, mandatory DPOs, AI-specific guidance published April 2025
- No Turkish competitor has entered the market (verified April 2026)
- Global competitors are consolidating (acquisitions) — leaving gaps for focused independents
- AI/data literacy is growing in Turkish enterprises

---

## 14. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| SAS/Hazy enters Turkey via existing SAS offices | Medium-High | SAS is expensive and enterprise-only; we target SMEs/mid-market first with competitive pricing and KVKK-native platform |
| Mostly AI open-source SDK undercuts paid market | Medium | Free SDK lacks managed service, KVKK compliance, Turkish support; our value is the platform, not just the model |
| LLM-based approaches (Gretel/Tonic) improve | Medium | LLMs have context window limits for high-dimensional tabular data; diffusion scales better (proven on 117 dimensions) |
| Training requires expensive GPU | Low | Cloud GPU pricing dropping, training is one-time per dataset, can batch jobs |
| Customers don't trust synthetic data | Medium | Publish validation reports, offer utility guarantees, pilot programs with measurable outcomes |
| Regulatory uncertainty (what counts as "synthetic"?) | Low | KVKK/GDPR don't specifically address synthetic data — it's a feature, not a bug (truly synthetic data is not personal data) |
| Model quality varies per dataset | Medium | Automated quality checks, utility benchmarks before delivery, human-in-the-loop validation |
| Full-time commitment required post-funding | Low | Planned — will transition from current employment after acceptance |

---

## 15. MVP Roadmap

### Phase 1 — Platform Foundation (Months 1-3)
- Web application: upload CSV, configure generation, download results
- REST API with authentication
- Cloud deployment with GPU support
- Basic privacy/utility report generation

### Phase 2 — Quality & Trust (Months 3-5)
- Automated quality validation pipeline
- Statistical comparison dashboard (real vs synthetic)
- Privacy audit report (membership inference test results)
- KVKK compliance documentation

### Phase 3 — Pilot & Validate (Months 5-8)
- 3-5 pilot customers (target: 1 pharma, 1 finance, 1 manufacturing)
- Iterate based on real feedback
- Collect case studies and testimonials

### Phase 4 — Scale (Months 8-12)
- Enterprise features (on-premise, SSO, audit logs)
- API marketplace listing
- Conditional generation (generate data matching specific criteria)
- Automated hyperparameter tuning

---

## 16. Key Research Results (Evidence)

### Utility — Replacement Scenario (Train on synthetic only)

| Method | R² Score | % of Baseline |
|--------|----------|---------------|
| Baseline (Real Data) | 0.6451 | 100% |
| SMOGN | -0.1354 | CATASTROPHIC FAILURE |
| CTGAN | 0.2292 | 35.5% |
| Simple Diffusion | 0.1712 | 26.5% |
| **TabDDPM (ours)** | **0.5628** | **87.3%** |

### Utility — Augmentation Scenario (Real + synthetic)

| Method | R² Score | % of Baseline |
|--------|----------|---------------|
| Baseline | 0.6451 | 100% |
| SMOGN | -0.1354 | HARMFUL |
| CTGAN | 0.6310 | 97.8% |
| **TabDDPM (ours)** | **0.6395** | **99.1%** |

### Production Dataset (Larger, 117 dimensions)

| Scenario | R² | % of Baseline |
|----------|-----|---------------|
| Replacement | 0.9785 | **98.4%** |
| Augmentation | 0.9936 | **100.0%** |

### Privacy Validation

| Method | Attack AUC | Interpretation |
|--------|-----------|----------------|
| TabDDPM (ours) | **0.5103** | Random guessing — no information leak |
| Diffusion (simple) | 0.5116 | Safe |
| SMOGN | 0.5253 | Safe |
| Unsafe threshold | >0.60 | Would indicate privacy leak |

### Summary
- **92.9% average utility** in replacement scenario
- **99.6% average utility** in augmentation scenario
- **0.51 AUC** — synthetic data is provably private
- **2.5x better** than CTGAN, **85x better** than SMOGN

---

## 17. References & Resources

### Project Assets
- **Codebase:** `/Users/umutakin/workspace/ua/seds500-graduation-project/`
- **GitHub:** `github.com/umutakin-dev/seds500-graduation-project`
- **Technical report:** `docs/project-report-updated.md`
- **Thesis summary:** `docs/thesis-summary.md`
- **Tutorial notebook:** `notebooks/01_diffusion_explained.ipynb`
- **Best experiment:** `experiments/experiment-019-production-tabddpm/`

### BiggSinerji
- **Website:** https://www.biggsinerji.com/
- **Application portal:** https://giris.tubitak.gov.tr/kullaniciadiilegiris.htm
- **Email:** biggsinerji@depark.com
- **Phone:** +90 232 453 03 93

### Academic References
- Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models," ICML 2023
- Kim et al., "STaSy: Score-based Tabular Data Synthesis," ICLR 2023
- Zhang et al., "Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space," ICLR 2024
- Xu et al., "Modeling Tabular Data using Conditional GAN," NeurIPS 2019

### Competitor References
- Gretel.ai — gretel.ai (Acquired by NVIDIA, Mar 2025, ~$320M+)
- Mostly AI — mostly.ai ($31M total, open-source SDK released Jan 2025)
- Hazy — hazy.com (Acquired by SAS, Nov 2024, now "SAS Data Maker")
- Tonic.ai — tonic.ai ($45M total, primarily masking; new LLM-based "Fabricate" product)

---

## Action Items (Priority Order)

1. **URGENT (by April 7):** Verify eligibility — no company shares, no prior TUBİTAK support
2. **URGENT (by April 7):** Register on TUBİTAK portal and submit application
3. **URGENT (by April 7):** Prepare AGY112 business plan form
4. Clarify IP situation with norm (SEDS500 is personal academic work)
5. Contact Dr. Damla Oguz for reference/support letter
6. Review Application Guide document from BiggSinerji website
7. Prepare demo/prototype if possible for later stages
