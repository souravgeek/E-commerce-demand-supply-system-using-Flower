# Research background and design justification

This document records the literature behind the project, what each source contributes, and how it
shaped a concrete design decision in the code. **Evidence note:** for the sources marked *(abstract
read)* I read the abstract / summary page, not the full paper; claims about them are limited to what
those pages state. Classic references marked *(standard reference)* are cited from general knowledge
of the field; please verify page/section details before quoting them in a report.

---

## 1. Problem statement

An e-commerce marketplace or retail group consists of many **stores / warehouses / sellers**. Each:

* observes its own sales, prices and promotions (commercially sensitive, sometimes regulated);
* must decide **how much stock to order** before demand is known (demand-supply matching);
* sees only a short, local history, so its own forecasting model is data-poor.

Two costs pull in opposite directions: *holding* stock (capital, storage, perishability) and
*stock-outs* (lost sales, unhappy customers). Forecast quality decides how well a store can balance them.

**Goal.** Build a distributed-analytics system in which stores jointly learn a demand-forecasting model
**without sharing raw sales data**, and measure the benefit where it matters, in *supply KPIs*
(fill rate, stock-outs, inventory, cost), not only in forecast error.

**Research questions answered by the experiments**

1. Does federated training beat what each store can learn alone (local-only)?
2. How close does it get to the unattainable ideal of pooling all data (centralized)?
3. Does the proximal term of FedProx help as stores become more different (non-IID)?
4. Do forecast gains translate into lower supply cost / higher fill rate?
5. Who benefits: the average store or the worst-off store?

---

## 2. Literature review

### 2.1 Federated learning foundations

| Source | Contribution | Used here for |
|---|---|---|
| McMahan, Moore, Ramage, Hampson, Agüera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017, arXiv:1602.05629 *(standard reference)* | Introduces **Federated Averaging (FedAvg)**: clients run several local SGD steps, the server averages the resulting models weighted by local data size. | The aggregation rule (`FedAvg` strategy; weighting key `num-examples`). |
| Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. *Federated Optimization in Heterogeneous Networks.* MLSys 2020, arXiv:1812.06127 *(abstract read)* | **FedProx**: a generalization / re-parametrization of FedAvg that adds a proximal term to each client's objective; the paper reports more robust and stable convergence than FedAvg on heterogeneous (non-IID) federated datasets and gives convergence guarantees. | The `proximal-mu` option: local loss + (μ/2)‖w − w_global‖². |
| Kairouz et al. *Advances and Open Problems in Federated Learning.* Found. & Trends in ML, 2021, arXiv:1912.04977 *(standard reference)* | Survey of FL: cross-silo vs cross-device settings, statistical heterogeneity, privacy, systems issues. | Framing: our stores are **cross-silo** participants (few, reliable, institutional). |
| Hsu, Qi, Brown. *Measuring the Effects of Non-IID Data Distribution for Federated Visual Classification.* 2019, arXiv:1909.06335 *(standard reference)* | Shows non-IID data degrades FedAvg and proposes controlled synthetic heterogeneity. | Motivation for a **tunable heterogeneity knob** `h` (0 = IID, 1 = very different stores). |
| Beutel et al. *Flower: A Friendly Federated Learning Research Framework.* arXiv:2007.14390 *(abstract read)* | Framework for FL research that scales to large simulated federations, is ML-framework agnostic, and lets experiments migrate from simulation to real devices. | The whole implementation (ClientApp / ServerApp / strategies / simulation runtime). |

### 2.2 Federated learning for demand forecasting and supply chains

| Source | What it shows | Relevance |
|---|---|---|
| Wang, Xie, Duan, Li, Sharma. *Federated Learning for Supply Chain Demand Forecasting.* Math. Problems in Engineering, 2022 *(abstract read)* | A retail supply-chain commodity demand-forecasting framework based on **vertical** FL, addressing data-security and privacy concerns of "new retail". | Establishes the application; ours is **horizontal** FL (same features, different stores), the natural fit for multi-store sales. |
| Turazza, Neri, Pietri, Butturi, Picone, Mamei. *Blockchain Federated Learning for Sustainable Retail: Reducing Waste through Collaborative Demand Forecasting.* IEEE ISCC 2025, arXiv:2602.04384 *(abstract and summary read)* | FL among grocery retailers to forecast perishable demand and reduce food waste; privacy barriers otherwise limit collaboration. Their algorithm is a FedAvg variant combined with SecAgg+ and global differential privacy with gradient clipping. | Confirms FedAvg as the baseline choice and **points to the privacy stack (secure aggregation + DP)** we list as future work and sketch in Section 5. |
| Long, Xu, Zheng, Brintrup. *PA-CFL: Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data.* arXiv:2503.12220 *(abstract read)* | Argues heterogeneity of retail data is the central difficulty and proposes clustered FL with privacy adaptation to balance accuracy and privacy. | Motivates the **non-IID experiments** and the "cluster similar stores" extension. |
| *Enhancing Supply Chain Demand Forecasting Through Gated Graph Neural Networks and Federated Learning Systems.* Informatica (journal) *(abstract read)* | Reports that, in their experiments, the FL model performs only about 4% worse than centralized training while avoiding data leaving its owner. | A published reference point for our expectation: **FL lands close to, but below, centralized**. |

**Gap and positioning.** Existing work focuses on a specific model (GNN, Transformer) or on privacy
machinery. This project contributes a **transparent, reproducible, end-to-end pipeline**: federated
forecasting *and* an explicit inventory simulation, with controlled heterogeneity and baselines. That
lets students *see* how a forecasting improvement becomes an operational (cost / service) improvement.

### 2.3 Privacy at production scale

The course slide deck on Gboard (Zheng Xu et al., *Federated Learning of Gboard Language Models with
Differential Privacy*, arXiv:2305.18465) shows the production recipe: federated learning for data
minimization, **secure aggregation**, **client update clipping**, **DP noise added to the aggregate**
(DP-FTRL), device-level privacy accounting, and the observation that **scale** (thousands of clients per
round) is what makes strong privacy affordable. Bonawitz et al., *Practical Secure Aggregation for
Privacy-Preserving Machine Learning* (ACM CCS 2017, *standard reference*) is the protocol behind
"the server only sees the sum of updates". Dwork & Roth, *The Algorithmic Foundations of Differential
Privacy* (2014, *standard reference*) is the formal background for (ε, δ)-DP.

Our baseline does **not** include these; Section 5 of the README and `docs/EXPLAINER.md` describe the
extension path, and the code isolates the hooks (client update, server aggregation) where they would go.

### 2.4 Forecasting and inventory theory

| Source | Used here for |
|---|---|
| Makridakis, Spiliotis, Assimakopoulos. *M5 accuracy competition: Results, findings, and conclusions.* Int. J. Forecasting, 2022 *(standard reference)* | Realistic retail demand benchmark; evidence that lag features, calendar effects, price and promotions drive demand. Our synthetic generator reproduces these drivers; real M5 data can be dropped in (see README). |
| Hyndman & Koehler. *Another look at measures of forecast accuracy.* IJF 2006 *(standard reference)* | Choosing **scale-free metrics**. We report **WAPE** (Σ\|error\| / Σ actual), which is robust to stores of very different size and well-defined with zero-sales days (unlike MAPE). |
| Silver, Pyke, Thomas. *Inventory and Production Management in Supply Chains*, 4th ed., CRC 2017 *(standard reference)*; Nahmias, *Production and Operations Analysis* | The **base-stock (order-up-to) policy** with safety stock `z·σ·√(L+1)`; the cost trade-off between holding and shortage. Implemented in `ecomfl/inventory.py`. |

---

## 3. From literature to design decisions

| Design decision | Why (source / reasoning) |
|---|---|
| Horizontal, cross-silo FL, one ClientApp per store | Same feature schema across stores; few reliable participants (Kairouz et al.; Wang et al. use vertical FL for a different data layout). |
| FedAvg as baseline | De facto standard (McMahan et al.), also the base of the retail FL work above. |
| FedProx as second method | Targets exactly the statistical heterogeneity that retail data exhibits (Li et al.; Long et al.). |
| Weighted aggregation by `num-examples` | FedAvg's definition; large stores influence the global model proportionally. |
| Tunable heterogeneity `h` and per-store reporting | Hsu et al.: heterogeneity must be controlled and measured; report **worst-store** results, not just means. |
| Temporal train/test split | Forecasting must not leak the future; random splits overstate accuracy. |
| `log1p(demand)` target + WAPE metric | Stores differ by an order of magnitude in scale; scale-free metrics (Hyndman & Koehler). |
| Local-only, centralized, naive baselines | Bracket the result: lower bound (alone), upper bound (pooled), trivial reference. The FL claim is only meaningful relative to these. |
| Base-stock simulation with sigma from **training** residuals | Evaluate forecasts by decision quality; avoid test leakage in the safety-stock estimate (inventory theory). |
| Flower | Course requirement; simulation runtime scales to many nodes and can move to real deployment (Beutel et al.). |

---

## 4. Known limitations (honest scope)

* **Synthetic data.** Generated to contain known structure (shared elasticity, weekly pattern, promo lift;
  store-specific scale, seasonality phase, mix). Conclusions are about the *method*, not about any real retailer.
  The loader interface (`StoreData`) allows real data (M5, Rossmann, Olist, Instacart) to be substituted.
* **Simulation, not deployment.** All nodes run on one machine. The server-side pooled benchmark would not
  exist in production (no central test set).
* **No formal privacy guarantee.** Weights can leak information (membership/inference attacks). Secure
  aggregation and DP, as in the Gboard and Turazza et al. work, are future extensions.
* **Single-echelon inventory model**, daily review, fixed lead time, no capacity / order-cost / substitution.
* **One-step-ahead forecasts** using true lagged sales; multi-step planning needs multi-horizon models.

---

## 5. References (BibTeX-ready identifiers)

1. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017. arXiv:1602.05629
2. T. Li et al., "Federated Optimization in Heterogeneous Networks," MLSys 2020. arXiv:1812.06127
3. D. J. Beutel et al., "Flower: A Friendly Federated Learning Research Framework," arXiv:2007.14390
4. P. Kairouz et al., "Advances and Open Problems in Federated Learning," Foundations and Trends in ML, 2021. arXiv:1912.04977
5. T.-M. H. Hsu, H. Qi, M. Brown, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification," 2019. arXiv:1909.06335
6. H. Wang, F. Xie, Q. Duan, J. Li, P. Sharma, "Federated Learning for Supply Chain Demand Forecasting," Mathematical Problems in Engineering, 2022.
7. F. Turazza et al., "Blockchain Federated Learning for Sustainable Retail: Reducing Waste through Collaborative Demand Forecasting," IEEE ISCC 2025. arXiv:2602.04384
8. Y. Long, L. Xu, G. Zheng, A. Brintrup, "PA-CFL: Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data," arXiv:2503.12220
9. Z. Xu et al., "Federated Learning of Gboard Language Models with Differential Privacy," 2023. arXiv:2305.18465
10. K. Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning," ACM CCS 2017.
11. C. Dwork, A. Roth, "The Algorithmic Foundations of Differential Privacy," 2014.
12. S. Makridakis, E. Spiliotis, V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions," Int. J. Forecasting, 2022.
13. R. Hyndman, A. Koehler, "Another look at measures of forecast accuracy," Int. J. Forecasting, 2006.
14. E. Silver, D. Pyke, D. Thomas, *Inventory and Production Management in Supply Chains*, 4th ed., CRC Press, 2017.
