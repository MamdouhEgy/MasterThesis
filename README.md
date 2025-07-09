#  Master Thesis: DDoS Attacks Detection and Mitigation in SDN using Deep Learning Approaches

This repository contains all supporting materials for my master's thesis conducted at the Communication Networks Group, Technische Universität Ilmenau, Department of Electrical Engineering and Information Technology.

**Thesis Duration**: December 2020 – June 2021  
**Author**: Mamdouh Muhammad  
**Supervision**: Prof. Dr. rer. nat. Jochen Seitz & M.Sc. Abdullah S. Alshra’a

---

##  Abstract

This work explores the detection and mitigation of **Distributed Denial-of-Service (DDoS) attacks** in **Software-Defined Networking (SDN)** environments using **deep learning techniques**, specifically a **VGG16-based Convolutional Neural Network (CNN)** architecture.

The research targets the SDN controller—a centralized yet vulnerable component in SDN architectures—and proposes a deep-learning-driven Intrusion Detection System (IDS) that transforms traffic flows into RGB image representations to leverage the power of CNNs for real-time classification.

---

##  Highlights

- **Dataset**: CICDDoS2019  
- **Preprocessing**:
  - Feature pruning, data cleaning, and class balancing
  - CSV-to-image transformation with 58×58 RGB matrix encoding
- **Model**:
  - Pretrained **VGG16 CNN** via transfer learning (ImageNet)
  - Tuned for SYN and UDP DDoS attacks
- **Evaluation Environment**:
  - SDN simulation using **Mininet** and **Ryu Controller**
  - Flow control via OpenFlow protocol actions (e.g., `OFPT_FLOW_MOD`)

---

##  Results

- VGG16-based IDS demonstrated superior accuracy, precision, recall, and F1-score compared to baseline CNN and classical ML models (ID3, RF, Naïve Bayes)
- Real-time controller-based mitigation is achieved through automated flow rule updates
- Effective against high-volume and low-rate DDoS attack variants

---


