# MaSE head contribution

**N (test):** 1250

## Global mixture weights

| Head | w |
|------|---|
| H0_VGG | 0.250 |
| H1_VGG+Res | 0.250 |
| H2_VGG+Res+Dense | 0.250 |
| H3_VGG+Res+Dense+MLP | 0.250 |

## Test accuracy

| Predictor | Acc |
|-----------|-----|
| H0_VGG alone | 0.8096 |
| H1_VGG+Res alone | 0.8384 |
| H2_VGG+Res+Dense alone | 0.8648 |
| H3_VGG+Res+Dense+MLP alone | 0.8624 |
| **Fused (MaSE)** | **0.8608** |
| Oracle (any head correct) | 0.8936 |
| Fusion gain vs best single head | -0.0040 |
| Oracle gap (room to improve) | 0.0328 |

## When fused is correct: head also voted true label

| Head | Share |
|------|-------|
| H0_VGG | 0.926 |
| H1_VGG+Res | 0.964 |
| H2_VGG+Res+Dense | 0.986 |
| H3_VGG+Res+Dense+MLP | 0.987 |

Mean head disagree fraction: 0.058

## Per-class accuracy (fused vs heads)

| Class | n | Fused | H0_VGG | H1_VGG+Res | H2_VGG+Res+Dense | H3_VGG+Res+Dense+MLP |
|-------|---|-------|---|---|---|---|
| cell periphery | 157 | 0.911 | 0.854 | 0.873 | 0.898 | 0.904 |
| cytoplasm | 128 | 0.922 | 0.883 | 0.891 | 0.945 | 0.930 |
| endosome | 69 | 0.551 | 0.464 | 0.478 | 0.551 | 0.522 |
| er | 176 | 0.920 | 0.875 | 0.915 | 0.926 | 0.920 |
| golgi | 38 | 0.632 | 0.579 | 0.632 | 0.684 | 0.684 |
| mitochondrion | 124 | 0.927 | 0.863 | 0.919 | 0.935 | 0.935 |
| nuclear periphery | 116 | 0.922 | 0.905 | 0.914 | 0.922 | 0.922 |
| nucleolus | 126 | 0.833 | 0.810 | 0.841 | 0.857 | 0.857 |
| nucleus | 163 | 0.914 | 0.847 | 0.883 | 0.902 | 0.920 |
| peroxisome | 16 | 0.688 | 0.500 | 0.688 | 0.688 | 0.688 |
| spindle pole | 78 | 0.731 | 0.744 | 0.718 | 0.679 | 0.667 |
| vacuole | 59 | 0.797 | 0.661 | 0.712 | 0.847 | 0.831 |
