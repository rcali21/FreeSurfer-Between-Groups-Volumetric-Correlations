#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from scipy.stats import spearmanr


wdir = os.getcwd()

print(wdir)
for f in os.listdir(wdir):
    if f.endswith(".xlsx"):
        print(f)

spreadsheet = input("Specify spreadsheet path::")

if os.path.exists(spreadsheet):
    print("")
    print("File path found.")

else:
    print("ERROR: File path not found.")


print(spreadsheet)


data = pd.read_excel(spreadsheet, sheet_name="T1_STATS_new")  # FILE INPUT
df = pd.DataFrame(
    data,
    columns=[
        "FS ID",
        "Left-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Left-Thalamus",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "3rd-Ventricle",
        "4th-Ventricle",
        "Brain-Stem",
        "Left-Hippocampus",
        "Left-Amygdala",
        "CSF",
        "Left-Accumbens-area",
        "Left-VentralDC",
        "Left-vessel",
        "Left-choroid-plexus",
        "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Vent",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
        "Right-Thalamus",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
        "Right-VentralDC",
        "Right-vessel",
        "Right-choroid-plexus",
        "5th-Ventricle",
        "WM-hypointensities",
        "Left-WM-hypointensities",
        "Right-WM-hypointensities",
        "non-WM-hypointensities",
        "Left-non-WM-hypointensities",
        "Right-non-WM-hypointensities",
        "Optic-Chiasm",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
        "BrainSegVol",
        "BrainSegVolNotVent",
        "lhCortexVol",
        "rhCortexVol",
        "CortexVol",
        "lhCerebralWhiteMatterVol",
        "rhCerebralWhiteMatterVol",
        "CerebralWhiteMatterVol",
        "SubCortGrayVol",
        "TotalGrayVol",
        "SupraTentorialVol",
        "SupraTentorialVolNotVent",
        "MaskVol",
        "BrainSegVol-to-eTIV",
        "MaskVol-to-eTIV",
        "lhSurfaceHoles",
        "rhSurfaceHoles",
        "SurfaceHoles",
        "EstimatedTotalIntraCranialVol",
    ],
)


data2 = pd.read_excel(spreadsheet, sheet_name="SYNTH_STATS_new")
df2 = pd.DataFrame(
    data2,
    columns=[
        "FS ID",
        "Left-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Left-Thalamus",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "3rd-Ventricle",
        "4th-Ventricle",
        "Brain-Stem",
        "Left-Hippocampus",
        "Left-Amygdala",
        "CSF",
        "Left-Accumbens-area",
        "Left-VentralDC",
        "Left-vessel",
        "Left-choroid-plexus",
        "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Vent",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
        "Right-Thalamus",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
        "Right-VentralDC",
        "Right-vessel",
        "Right-choroid-plexus",
        "5th-Ventricle",
        "WM-hypointensities",
        "Left-WM-hypointensities",
        "Right-WM-hypointensities",
        "non-WM-hypointensities",
        "Left-non-WM-hypointensities",
        "Right-non-WM-hypointensities",
        "Optic-Chiasm",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
        "CC_total",
        "BrainSegVol",
        "BrainSegVolNotVent",
        "lhCortexVol",
        "rhCortexVol",
        "CortexVol",
        "lhCerebralWhiteMatterVol",
        "rhCerebralWhiteMatterVol",
        "CerebralWhiteMatterVol",
        "SubCortGrayVol",
        "TotalGrayVol",
        "SupraTentorialVol",
        "SupraTentorialVolNotVent",
        "MaskVol",
        "BrainSegVol-to-eTIV",
        "MaskVol-to-eTIV",
        "lhSurfaceHoles",
        "rhSurfaceHoles",
        "SurfaceHoles",
        "EstimatedTotalIntraCranialVol",
    ],
)


# THIS IS FOR ADDING LATERALIZED REGIONS INTO ONE FOR GROUND-TRUTH
Cerebral_White_Matter = df["lhCerebralWhiteMatterVol"] + df["rhCerebralWhiteMatterVol"]
Cerebellar_cortex = df["Left-Cerebellum-Cortex"] + df["Right-Cerebellum-Cortex"]
Corpus_callosum = (
    df["CC_Posterior"]
    + df["CC_Mid_Posterior"]
    + df["CC_Central"]
    + df["CC_Mid_Anterior"]
    + df["CC_Anterior"]
)
Hippocampus = df["Left-Hippocampus"] + df["Right-Hippocampus"]
Putamen = df["Left-Putamen"] + df["Right-Putamen"]
Caudate = df["Left-Caudate"] + df["Right-Caudate"]
Globus_Pallidus = df["Left-Pallidum"] + df["Right-Pallidum"]
Cerebellar_White_matter = (
    df["Left-Cerebellum-White-Matter"] + df["Right-Cerebellum-White-Matter"]
)
# THIS IS FOR ADDING LATERALIZED REGIONS INTO ONE FOR SYNTH
Cerebral_White_Matter2 = (
    df2["lhCerebralWhiteMatterVol"] + df2["rhCerebralWhiteMatterVol"]
)
Cerebellar_cortex2 = df2["Left-Cerebellum-Cortex"] + df2["Right-Cerebellum-Cortex"]
Corpus_callosum2 = (
    df2["CC_Posterior"]
    + df2["CC_Mid_Posterior"]
    + df2["CC_Central"]
    + df2["CC_Mid_Anterior"]
    + df2["CC_Anterior"]
)
Hippocampus2 = df2["Left-Hippocampus"] + df2["Right-Hippocampus"]
Putamen2 = df2["Left-Putamen"] + df2["Right-Putamen"]
Caudate2 = df2["Left-Caudate"] + df2["Right-Caudate"]
Globus_Pallidus2 = df2["Left-Pallidum"] + df2["Right-Pallidum"]
Cerebellar_White_matter2 = (
    df2["Left-Cerebellum-White-Matter"] + df2["Right-Cerebellum-White-Matter"]
)
coef, p = spearmanr(Cerebral_White_Matter, Cerebral_White_Matter2, nan_policy="omit")


ground_truth_list = []
# ground_truth_list.append(Cerebral_White_Matter)
# ground_truth_list.append(Cerebellar_cortex)
# ground_truth_list.append(Corpus_callosum)
# ground_truth_list.append(Hippocampus)
# ground_truth_list.append(Putamen)
# ground_truth_list.append(Caudate)
# ground_truth_list.append(Globus_Pallidus)
# ground_truth_list.append(Cerebellar_White_matter)


synth_list = []
# synth_list.append(Cerebral_White_Matter2)
# synth_list.append(Cerebellar_cortex2)
# synth_list.append(Corpus_callosum2)
# synth_list.append(Hippocampus2)
# synth_list.append(Putamen2)
# synth_list.append(Caudate2)
# synth_list.append(Globus_Pallidus2)
# synth_list.append(Cerebellar_White_matter2)


# for q in ground_truth_list:
#     for l in synth_list:
#         coef, p = spearmanr(q, l, nan_policy='omit')
#         print('Cerebellar_White_matter Mean Ground-Truth:', (round(q.mean(),2)))
#         print('Mean Synth:', (round(l.mean(),2)))
#         print('Standard Deviation Ground-Truth:', round(q.std(),2))
#         print('Standard Deviation Synth:', round(l.std(),2))
#         print('Coef:', coef)
#         print('P-value:', "%.10f" % p)


# print(w, 'Coef:', coef)
# print(w, 'P-value:', p)


# print(coef)
# print('P-value:', "%.10f" % p)
# print(Cerebral_White_Matter.mean())
# print(Cerebral_White_Matter.std())
# print(Cerebral_White_Matter2.mean())
# print(Cerebral_White_Matter2.std())

regions = [
    "CortexVol",
    "SubCortGrayVol",
    "Brain-Stem",
    "TotalGrayVol",
    "CerebralWhiteMatterVol",
]


for i in regions:

    coef, p = spearmanr(df[i], df2[i], nan_policy="omit")

    print(i, "Standard deviation GROUND-TRUTH:", df[i].std())
    print(i, "Mean GROUND-TRUTH:", df[i].mean())
    print(i, "Standard deviation SYNTH:", df2[i].std())
    print(i, "Mean SYNTH:", df2[i].mean())

    # ---------------- df = Ground-Truth
    # ---------------- df = Synth

    print(i, "P-value:", "%.10f" % p)

    print(i, "Spearmans correlation coefficient: %.3f" % coef)

coef, p = spearmanr(df["Brain-Stem"], df2["Brain-Stem"], nan_policy="omit")
print(coef, p)
# alpha = 0.05
# if p > alpha:
#     print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
# else:
#     print('Samples are correlated (reject H0) p=%.3f' % p)

# print("Standard deviation:", df['SubCortGrayVol'].std())
# print("Mean:", df['SubCortGrayVol'].mean())
# print("Standard deviation:", df2['SubCortGrayVol'].std())
# print("Mean:", df2['SubCortGrayVol'].mean())


# Bonferroni with 11 tests

# x = 0.05/11
# print(x)


data = pd.read_excel(spreadsheet, sheet_name="T1_STATS_new")  # FILE INPUT
df = pd.DataFrame(
    data,
    columns=[
        "FS ID",
        "Left-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Left-Thalamus",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "3rd-Ventricle",
        "4th-Ventricle",
        "Brain-Stem",
        "Left-Hippocampus",
        "Left-Amygdala",
        "CSF",
        "Left-Accumbens-area",
        "Left-VentralDC",
        "Left-vessel",
        "Left-choroid-plexus",
        "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Vent",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
        "Right-Thalamus",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
        "Right-VentralDC",
        "Right-vessel",
        "Right-choroid-plexus",
        "5th-Ventricle",
        "WM-hypointensities",
        "Left-WM-hypointensities",
        "Right-WM-hypointensities",
        "non-WM-hypointensities",
        "Left-non-WM-hypointensities",
        "Right-non-WM-hypointensities",
        "Optic-Chiasm",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
        "CC_total",
        "BrainSegVol",
        "BrainSegVolNotVent",
        "lhCortexVol",
        "rhCortexVol",
        "CortexVol",
        "lhCerebralWhiteMatterVol",
        "rhCerebralWhiteMatterVol",
        "CerebralWhiteMatterVol",
        "SubCortGrayVol",
        "TotalGrayVol",
        "SupraTentorialVol",
        "SupraTentorialVolNotVent",
        "MaskVol",
        "BrainSegVol-to-eTIV",
        "MaskVol-to-eTIV",
        "lhSurfaceHoles",
        "rhSurfaceHoles",
        "SurfaceHoles",
        "EstimatedTotalIntraCranialVol",
    ],
)  # LIST


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
l = df.corrwith(df2, method="spearman", axis=0)
print(l)


X = df2["CC_total"]
Y = df["CC_total"]


def best_fit(X, Y):  # User-defined function

    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)

    numer = sum(xi * yi for xi, yi in zip(X, Y)) - n * xbar * ybar
    denum = sum(xi ** 2 for xi in X) - n * xbar ** 2

    b = numer / denum
    a = ybar - b * xbar

    print("best fit line:\ny = {:.2f} + {:.2f}x".format(a, b))

    return a, b


# solution

a, b = best_fit(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ticklabel_format(style="plain")
yfit = [a + b * xi for xi in X]
ax.set_xlabel("Ground-Truth", color="red")
ax.set_ylabel("Synthetic", color="blue")
plt.scatter(X, Y, c="red")
plt.scatter(Y, X, c="blue")
plt.plot(X, yfit)
plt.grid()
plt.title("CC_total")
plt.savefig("Correlation.png")  # Output
plt.show()
