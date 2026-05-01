# COMMUNITY DETECTION 
# Community Detection: Label Propagation Algorithm

This project explores how **Community Detection** algorithms can identify clusters in a network even when inter-group connections ("noise") exist.

### **Algorithm: Label Propagation**
We implement a custom `LabelPropagation` class where:
1. Every node starts with a unique label.
2. Nodes iteratively adopt the label with the **highest total weight** from their neighbors.
3. The process converges when no nodes change labels.

### **Network Structure**
*   **Nodes**: 10-node network.
*   **Weights**: Internal group links have a weight of **5**, while inter-community "bridges" have a weight of **1**.
*   **Result**: The model successfully identifies two distinct communities despite the weak bridges.

### **Dependencies**
```bash
pip install numpy networkx matplotlib
