import numpy as np
import sys
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities")
from t_a_Manipulation import replicates_to_featurematrix




#------------------------------------------------------------
#Loading in our files of interest and creating feature matrix
#------------------------------------------------------------
replicate_frames = (([80] * 20) + ([160] * 10)) * 2

#Load in arrays and make them a list then concatenate them
redone_CCU_GCU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',allow_pickle=True)
redone_CCU_CGU_fulltraj=np.load('/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',allow_pickle=True)

#------------------------------------------------------------
#doing it
#------------------------------------------------------------
import numpy as np
from grakel import GraphKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from grakel import GraphKernel

# === Optional: helper script path ===
sys.path.append("/zfshomes/lperez/final_thesis_scripts/pypure/utilities")
from t_a_Manipulation import replicates_to_featurematrix

# === Load Data ===
# Assume each is (2300, 230, 300)
redone_CCU_GCU_fulltraj = np.load(
    '/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_GCU_Trajectory_array.npy',
    allow_pickle=True)
redone_CCU_CGU_fulltraj = np.load(
    '/zfshomes/lperez/final_thesis_data/redone_unrestrained_CCU_CGU_Trajectory_array.npy',
    allow_pickle=True)

# Remove padding (if applicable)
X_A = redone_CCU_GCU_fulltraj[:, 1:, 1:]
X_B = redone_CCU_CGU_fulltraj[:, 1:, 1:]

# === Convert to GraKeL format ===
def convert_to_grakel_format(adj_matrices):
    graphs = []
    for mat in adj_matrices:
        if mat.shape[0] == mat.shape[1]:  # Ensure square adjacency
            graphs.append(mat)
    return graphs

graphs_A = convert_to_grakel_format(X_A)
graphs_B = convert_to_grakel_format(X_B)

# === Combine sets and compute full kernel matrix ===
graphs_all = graphs_A + graphs_B
gk = GraphKernel(kernel={"name": "shortest_path"}, normalize=True)
K_all = gk.fit_transform(graphs_all)  # shape: (4600, 4600)

labels = np.array(
    [f"{i+1}a" for i in range(len(graphs_A))] +
    [f"{i+1}b" for i in range(len(graphs_B))]
)
# === Save the full kernel matrix and labels ===
os.makedirs("kernel_outputs", exist_ok=True)
np.save("kernel_outputs/K_all.npy", K_all)
np.save("kernel_outputs/labels.npy", labels)
print("✅ Saved joint kernel matrix and labels.")

# === Visualize combined graph space via PCA ===
pca = PCA(n_components=2)
X_2d = pca.fit_transform(K_all)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='plasma', s=10)
plt.title("📊 Joint PCA of Graph Similarity Space (A vs B)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("kernel_outputs/embedding_joint_pca_plot.png", dpi=300)
plt.show()

print("✅ PCA plot saved with plasma colormap.")

# === Optional: Slice K_all for A vs A, B vs B, A vs B ===
n = len(graphs_A)
K_AA = K_all[:n, :n]
K_BB = K_all[n:, n:]
K_AB = K_all[n:, :n]
K_BA = K_all[:n, n:]

np.save("kernel_outputs/K_AA.npy", K_AA)
np.save("kernel_outputs/K_BB.npy", K_BB)
np.save("kernel_outputs/K_AB.npy", K_AB)
np.save("kernel_outputs/K_BA.npy", K_BA)

print("✅ Sliced kernel matrices saved.")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# === Train/test split indices ===
# Since you want to test A vs B, we’ll use half for training, half for testing
n_total = len(labels)
n = len(graphs_A)

# Use half of each class for training
train_idx = np.concatenate([np.arange(n//2), n + np.arange(n//2)])
test_idx = np.concatenate([np.arange(n//2, n), n + np.arange(n//2, n)])

K_train = K_all[np.ix_(train_idx, train_idx)]
K_test = K_all[np.ix_(test_idx, train_idx)]
y_train = labels[train_idx]
y_test = labels[test_idx]

# === Train SVM on precomputed kernel ===
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# === Evaluate and save accuracy ===
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy (A vs B): {acc:.4f}")

# Save accuracy score
with open("kernel_outputs/accuracy.txt", "w") as f:
    f.write(f"Accuracy (SVM on Shortest Path Kernel, A vs B): {acc:.4f}\n")

print("✅ Accuracy saved to kernel_outputs/accuracy.txt")
