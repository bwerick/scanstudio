# ScanStudio: Improving and Evaluating a Video-Based Book Digitization Workflow

**Rohan Reddy Boddu**

Honors thesis draft | Arizona State University | November 2026

**Committee information and final title: confirm with committee**

> Draft for committee review. This document describes an initial recorded-video case study, not a completed quantitative validation. Confirm individual contributions, citations, committee information, formatting, and any AI-assistance disclosure with the thesis director before submission.

## Abstract

Digitizing a bound book from video requires selecting usable views of pages while the operator turns them. Duplicate captures, page-turn motion, hands, and imperfect cropping can lead to extra or unreadable pages in the resulting document. This thesis examines ScanStudio, an open-source workflow that extracts representative frames from a book-scanning video, supports human review, and assembles page images into a PDF. The contribution considered here is an advisory image-based duplicate flagging pass placed between keyframe extraction and human review, together with an initial case study of the end-to-end workflow. The pass compares nearby grayscale previews using a difference hash and normalized pixel difference; it does not automatically delete frames. In one 94.5-second, 720p recording, the pipeline selected 18 candidate keyframes and the detector flagged none. During review, the operator removed three mid-turn frames. The remaining 15 spreads produced a 30-page PDF. Visual inspection of the output found repeated content corresponding to one repeated spread, hand obstruction, a split cover, and clipping. These observations show both that the workflow can produce a document and that the current heuristic and review process do not guarantee a complete, clean scan. Precision and recall were not measured because a verified candidate-level reference set was not available. The thesis concludes with a plan for preserving source images, labeling candidate frames, improving duplicate comparisons, and evaluating results on separate recordings.

## 1. Introduction

Printed books remain useful sources of knowledge, but their physical form limits how easily their contents can be searched, shared, and analyzed. Photographing an open book avoids separating its pages from the binding and makes use of widely available cameras. Yet the apparent simplicity of recording a page-turning session hides a sequence of decisions: determine when a page has changed, select a sharp frame, distinguish repeated captures from unique pages, remove unusable images, and retain all page content when cropping and splitting the spread.

ScanStudio addresses this workflow by processing video of a book and providing review tools before assembling a PDF [1]. Its existing code supports both live capture and offline processing, followed by image review and document generation. This thesis focuses on the offline recorded-video path and on a specific review burden: repeated or poor candidate frames. A duplicate may result when the same spread is selected more than once, while a frame taken in the middle of a page turn may be unusable for a different reason. Hands can obscure text even when the page itself is otherwise sharp. These cases require separate treatment because automatically discarding the wrong image may remove the only available capture of a page.

The research questions are: (1) how can a lightweight image-only pass identify likely duplicate captures before review, and (2) what does an initial recorded-video run reveal about the opportunities and limits of this approach? The work adds a conservative advisory pass, documents a compatibility repair needed by an existing review interface, and analyzes one completed scan. It does not claim a validated duplicate-removal system or an automatic solution for occlusion or missing-page coverage.

## 2. Background and related work

### 2.1 Camera-based digitization

Document capture ranges from flatbed scanning to camera-based photography and video. A flatbed constrains the imaging surface but requires physical placement; overhead camera capture allows an operator to turn a bound book under a fixed camera. The latter introduces perspective, curved pages, uneven lighting, and hands into the images. Ma et al. describe the difficulty of rectifying curved and folded documents and note that some approaches use additional geometric hardware, while others infer a correction from images alone [2]. The practical objective in ScanStudio is a workflow usable with ordinary capture equipment and human review; this case study does not measure hardware cost or preservation outcomes.

### 2.2 Video capture and page selection

Video-based scanning reduces the need to trigger an individual photograph for every page, but it must identify when one page or spread ends and another begins. Wigington proposes a learned on-device approach for multi-page scanning with a handheld phone, explicitly addressing camera movement and instability [3]. ScanStudio's recorded-video baseline instead uses a motion signal and selects sharp candidates in detected intervals. These systems make different assumptions about capture conditions and use different detection methods. This thesis does not compare their accuracy or claim that ScanStudio is unique in using video.

Frame selection and duplicate identification are related but distinct. A page-turn detector can divide video into too many or too few intervals; even a sharp selected frame may repeat a previous spread. Conversely, a duplicate comparison should avoid confusing two distinct pages with similar layouts. A human review step offers a way to inspect uncertain results, but its workload and error rate need measurement.

### 2.3 Image correction and OCR

Cropping and perspective correction can isolate the page and map a quadrilateral view to a rectangular image. OpenCV documents homography-based perspective correction [4]. However, a planar transform cannot in general flatten every locally curved surface. DocUNet instead learns a deformation mapping for document unwarping and reports evaluation on a benchmark of real images [2]. More elaborate restoration may improve warped pages, but this thesis does not implement or benchmark DocUNet.

Optical character recognition converts images into machine-readable text, but recognition quality depends on the input. Tesseract's documentation discusses factors such as resolution, skew, and image preprocessing [5]. OCR could support duplicate comparison using extracted words or identify page numbers for a coverage check. It also introduces recognition errors and may fail on graphics, blank pages, or obscured text. The present duplicate pass uses images only; no OCR accuracy or searchable-PDF result is claimed for the tested output.

## 3. System and contribution

### 3.1 Existing recorded-video workflow

The recorded-video pipeline first computes a motion signal from consecutive frames and detects peaks associated with page-turn activity. Detected boundaries define intervals in which candidate keyframes are selected. The selection stage favors low-motion candidates and uses a sharpness measure to choose among them. The output is a set of representative frames and metadata. The exact behavior is dependent on the capture conditions and detector settings: a pause during a turn or repeated capture can still produce an undesired candidate [1].

The review stage allows the operator to inspect candidates and remove unsuitable frames. The subsequent pipeline crops retained spreads, splits double-page images into pages, permits page review, and builds a PDF. In the tested browser workflow, these review actions were manual. An additional repair restored crop helpers imported by the native keyframe-review script after an import error prevented that interface from opening. The remote experiment was completed through browser review; it does not by itself validate every native interface configuration.

### 3.2 Advisory duplicate detection

The implemented Phase 3.5 runs after keyframe selection and before review. Each image is loaded in grayscale, resized to 96 by 96 pixels, and lightly blurred. The system calculates a 64-bit difference hash and a normalized mean absolute pixel difference for a pair of preview images. For previews A and B, the normalized difference is the mean of the absolute pixel differences divided by 255. The hash distance counts differing hash bits.

With the default configuration, each candidate is compared against the immediately preceding candidate. The later frame is flagged if its hash distance is at most 4 and its normalized pixel difference is at most 0.02. Both conditions must hold. These are initial heuristic settings, not calibrated thresholds. The program writes `frame_quality.json` and adds flags to keyframe metadata; it does not remove frames. A native review interface displays flags, while the browser interface used for the case study did not yet display them. The absence of flags in the November run, however, is a detection result and would not be fixed merely by displaying the warnings.

The design is inexpensive and avoids requiring OCR or a trained model. Its limitations follow from comparing the whole low-resolution image without explicit alignment or page-region masking. A moving hand, camera displacement, lighting changes, or a page-turn frame between two views can prevent a match. Similar layouts may also lead to false matches. The experiment below provides a concrete failure case, not a complete characterization of these risks.

## 4. Methods

### 4.1 Material and procedure

One recording of pages from a January 2024 issue of *Communications of the ACM* was processed on November 9, 2026. The source was 1280 by 720 pixels, 30 frames per second, and 2,835 frames long (94.5 seconds). The run used ScanStudio commit `2c0ed7f` on a remote Linux computer. The operator connected through Tailscale and used an SSH port forward to access the browser review interface from a Mac. This was a recorded-video test, not an evaluation of live capture.

The system computed the motion signal, detected candidate intervals, selected keyframes, and ran the Phase 3.5 duplicate pass. The operator then reviewed the candidates and removed frames captured during page turns. The retained images were cropped and split, reviewed, and assembled into a PDF. The terminal log records stage counts and source resolution. The operator reported that the three discarded frames were caught mid-turn. The final PDF was inspected visually page by page; PDF positions in this thesis refer to the output file, not necessarily the magazine's printed page numbers.

### 4.2 What this evaluation can and cannot measure

The case study reports observed counts and visible output defects. It is not a controlled comparison of algorithm variants. The original 18 candidate images and corresponding reference labels were not assembled into a verified evaluation set for this draft. Consequently, detector precision and recall, false-positive rates, missed source pages, and generalization to other books cannot be calculated from the available evidence. The 30-page PDF alone cannot establish which source pages were never captured or why a particular crop failed.

## 5. Results

### 5.1 Processing counts

| Stage | Observation |
| --- | --- |
| Input | 94.5 seconds; 1280 × 720; 30 fps |
| Detected motion peaks | 17 |
| Selected candidates | 18 keyframes |
| Phase 3.5 duplicate flags | 0 |
| Manual removals | 3 mid-turn frames |
| Retained images | 15 spreads; zero covers tagged |
| Final output | 30 PDF pages; approximately 2.1 MB |

The fraction of selected candidates manually removed was 3/18, or 16.7%. This is a count of operator intervention in one run, not a duplicate-detection error rate. Because the removed frames were identified as mid-turn, they should not be counted automatically as duplicates. Likewise, the 30-page output does not by itself prove that 30 distinct and complete source pages were captured.

### 5.2 Visual observations

Output positions 7 and 9 show the same publication-information page; positions 8 and 10 show the same facing editorial page. Together, these are consistent with one repeated spread surviving review and export. The two captures differ in hand position and framing. The detector's zero flags therefore cannot be interpreted as proof that no duplicate content existed. The final output alone does not establish all reasons the comparison failed.

Hands obscure text in several images, especially output positions 7 and 20. The three manually deleted mid-turn frames did not eliminate all obstruction in retained frames. The first cover image was split across output positions 1 and 2, and both halves retain substantial table background. The run log records zero tagged covers, so this example does not demonstrate that a tagged cover was ignored. Position 16 has clipped text along the left edge; inspecting the source frame and crop and split geometry is necessary to identify the responsible stage.

The tested PDF contains page images without extractable text. Its small text is constrained by the 720p recording and framing. Increasing PDF export dimensions cannot recreate detail absent from the source images. A higher-resolution capture is an appropriate future test, but no such comparison was completed here.

## 6. Discussion

The remote run demonstrates that the recorded-video workflow can complete end to end on this example and produce a reviewable PDF. It also highlights the gap between producing a PDF and producing a clean, complete digital document. Motion segmentation, keyframe quality, duplication, geometry, and page coverage each contribute different kinds of error; a single frame-similarity score cannot establish all of them.

The advisory detector returned no duplicate flags despite repeated content surviving in the output. This outcome motivates an improved comparison, but does not determine which threshold or method would generalize. Page-region detection and geometric alignment may reduce sensitivity to background or camera movement. Comparing more than one earlier candidate may help when other frames intervene, but increases the chance of false matches and must be evaluated. OCR-based comparison is another option, although it would need to handle recognition failures, non-text pages, and similar text.

The test also points to the role of user interface design. Warnings are useful only when visible in the workflow the operator actually uses; the browser review should display any flags and the matching frame. Human decisions and original images should be preserved to make the process auditable. The current crop workflow can overwrite working images, so separating source frames from transformed outputs would support safe reruns and reproducibility.

Several conclusions remain tentative. This is one recording, and the sample has not been independently labeled. There is no benchmark of alternative thresholds, no comparison with an OCR-based approach, no test of live-mode usability, and no demonstrated automatic detection of hands or missing pages. The three earlier recordings described in project meetings appeared qualitatively cleaner on later attempts, but they were not controlled experiments and are not used as evidence of an algorithmic improvement.

## 7. Future work

The next experiment should preserve all candidate frames and label the identity of each spread independently of the detector output. A reference label should identify whether a candidate repeats an earlier spread; separate fields should describe mid-turn motion, obstruction, and the preferred usable capture. Duplicate precision is TP/(TP+FP), and duplicate recall is TP/(TP+FN), where TP is a correctly flagged repeated candidate, FP a unique candidate incorrectly flagged, and FN a repeated candidate not flagged. Ratios with zero denominators should be reported as not applicable. Page-level duplicate pairs derived from one repeated spread must not be counted as two separate duplicate-spread detections.

After baseline labels are established, compare the original heuristic with page-focused or aligned comparisons, and tune thresholds using development recordings. Evaluate the selected method on separate recordings not used for tuning. Report manual review time and corrections alongside precision and recall, and investigate capture resolution and framing as separate factors. A representative test set should include similar but distinct pages, repeated pages with changed hand positions, and unusable frames.

Occlusion detection, OCR-assisted comparisons, and missing-page coverage are possible extensions. They should be evaluated as separate tasks with their own reference labels. For example, a page-number sequence could reveal some gaps, but not every page has a recognizable number, and an OCR failure should not be treated as proof that a page is missing. If future research involves studies with human participants, the director and ASU research compliance staff should determine applicable review requirements before recruitment.

## 8. Conclusion

This thesis described an image-based advisory duplicate pass in ScanStudio and examined one complete video-to-PDF run. The pass is computationally simple and keeps the operator in control, but it missed repeated content in the inspected output. Manual review removed three mid-turn frames, while other issues remained. The main outcome is a documented baseline and a concrete evaluation plan: preserve original evidence, make warnings visible, label duplicate and quality failures separately, and test improvements on independent recordings. Further ScanStudio research can continue beyond this honors thesis without claiming that these future steps have already been completed.

## References

[1] ScanStudio contributors. *ScanStudio source code*, commit `2c0ed7f`. https://github.com/bwerick/scanstudio/tree/2c0ed7f

[2] K. Ma, Z. Shu, X. Bai, J. Wang, and D. Samaras. “DocUNet: Document Image Unwarping via a Stacked U-Net.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 4700–4709, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Ma_DocUNet_Document_Image_CVPR_2018_paper.html

[3] C. Wigington. “Handheld Video Document Scanning: A Robust On-Device Model for Multi-Page Document Scanning.” arXiv:2411.00576, 2024. https://arxiv.org/abs/2411.00576

[4] OpenCV contributors. “Basic Concepts of the Homography Explained with Code.” OpenCV documentation. https://docs.opencv.org/4.4.0/d9/dab/tutorial_homography.html

[5] Tesseract contributors. “Improving the Quality of the Output.” Tesseract documentation. https://github.com/tesseract-ocr/tessdoc/blob/main/ImproveQuality.md

## Evidence and author verification

The recorded-video counts are drawn from the November 9 terminal transcript and the final PDF completion message. Output observations were made by inspecting `ScanStudio_remote_run_2026-11-09.pdf`. The project proposal and supervisor meeting notes provide context. The author should verify the account of individual contributions, committee roles, bibliographic style, and institutional disclosure requirements. This draft was prepared with AI assistance and requires author and committee review.
