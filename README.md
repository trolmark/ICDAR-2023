# ICDAR-2023
# Competition on Reading the Seal Title

This competition consists of two main tasks:

1. Seal title text detection
2. End-to-end seal title recognition

The datasets cover the most common classes of seals:

- Circle/Ellipse shapes: This type of seals are commonly existing in official seals, invoice seals, contract seals, and bank seals.
- Rectangle shapes: This type of seals are commonly seen in driving licenses, corporate seals, and medical bills.
- Triangle shapes: This type of seals are seen in bank receipts and financial occasions. This type is uncommon seal and has a small amount of data.

## Seal title text detection solution

This method formulates the text detection task as one of identifying particular keypoints which are subsequently utilized to delineate the boundaries of curved text regions. A bottom-up approach is employed in this technique, similar to that commonly used for pose-estimation tasks, where all keypoint locations in the image are determined and subsequently grouped together according to geometric rules to form a polygon which defines the curved text area.

## End-to-end seal title recognition solution

The solution to the second task is an implementation of the arbitrary orientation network (AON) to directly capture the deep features of irregular texts, which are combined into an attention-based decoder to generate character sequence. The whole network can be trained end-to-end by using only images and word-level annotations.

Read the paper on AON here: https://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_AON_Towards_Arbitrarily-Oriented_CVPR_2018_paper.pdf





