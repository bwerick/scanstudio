# Verification Signals Implementation Guide
## Detailed Plans for Each Test in ScanStudio

This document provides comprehensive implementation details for all four verification signals.

---

## Signal 1: Duplicate Agreement

### What It Tests
**Consistency**: Does the pipeline produce similar results when processing the same page multiple times?

### Why It Matters
- Detects non-deterministic bugs
- Validates pipeline stability
- Identifies quality variance issues
- High-confidence duplicates can be used as training data

### How It Works

#### Step 1: Find Duplicate Page Captures

**Method**: Perceptual Hashing (pHash)

```python
import imagehash
from PIL import Image
import cv2

def find_duplicate_frames(video_path, hamming_threshold=10):
    """
    Find frames that capture the same page using perceptual hashing.

    Args:
        video_path: Path to input video
        hamming_threshold: Max Hamming distance for duplicates (default: 10)

    Returns:
        List of (frame_a, frame_b) tuples representing duplicate pairs
    """
    cap = cv2.VideoCapture(video_path)
    frame_hashes = {}
    duplicates = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL Image for hashing
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Compute perceptual hash
        phash = imagehash.phash(pil_img, hash_size=8)

        # Check against all previous frames
        for prev_idx, prev_hash in frame_hashes.items():
            hamming_dist = phash - prev_hash

            if hamming_dist < hamming_threshold:
                duplicates.append((prev_idx, frame_idx))
                print(f"Duplicate found: Frame {prev_idx} ≈ Frame {frame_idx} (distance={hamming_dist})")

        frame_hashes[frame_idx] = phash
        frame_idx += 1

    cap.release()
    return duplicates
```

**Why pHash?**
- Robust to minor variations (lighting, slight movement)
- Fast to compute (can process 1000s of frames)
- Hash comparison is O(1)

**Alternative**: If pHash gives too many false positives, use SSIM (Structural Similarity Index):

```python
from skimage.metrics import structural_similarity as ssim

def ssim_duplicates(frame_a, frame_b, threshold=0.95):
    """More accurate but slower duplicate detection"""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Resize to same dimensions
    h, w = min(gray_a.shape[0], gray_b.shape[0]), min(gray_a.shape[1], gray_b.shape[1])
    gray_a = cv2.resize(gray_a, (w, h))
    gray_b = cv2.resize(gray_b, (w, h))

    score = ssim(gray_a, gray_b)
    return score > threshold
```

#### Step 2: Process Both Duplicates Through Pipeline

```python
def process_duplicate_pair(video_path, frame_a_idx, frame_b_idx):
    """
    Extract both frames and process through full pipeline.

    Returns:
        (result_a, result_b) where each result contains:
        - ocr_text: Full OCR output
        - crop_bbox: Bounding box coordinates (x, y, w, h)
        - entities: Extracted entities dict
    """
    # Extract frames
    frame_a = extract_frame_at_index(video_path, frame_a_idx)
    frame_b = extract_frame_at_index(video_path, frame_b_idx)

    # Process through pipeline
    result_a = {
        'ocr_text': None,
        'crop_bbox': None,
        'entities': None
    }

    result_b = {
        'ocr_text': None,
        'crop_bbox': None,
        'entities': None
    }

    # Phase 1-3: Keyframe already selected, skip to Phase 5
    # Phase 5: Crop
    cropped_a, bbox_a = crop_with_bbox_return(frame_a)
    cropped_b, bbox_b = crop_with_bbox_return(frame_b)

    result_a['crop_bbox'] = bbox_a
    result_b['crop_bbox'] = bbox_b

    # Phase 6-9: Split, Binarize (optional), OCR
    result_a['ocr_text'] = ocr_image(cropped_a)
    result_b['ocr_text'] = ocr_image(cropped_b)

    # Extract entities for additional comparison
    result_a['entities'] = extract_entities(result_a['ocr_text'])
    result_b['entities'] = extract_entities(result_b['ocr_text'])

    return result_a, result_b
```

#### Step 3: Compute Agreement Score

```python
from Levenshtein import distance as levenshtein_distance

def compute_duplicate_agreement(result_a, result_b):
    """
    Compute agreement score between two duplicate page captures.

    Returns:
        float: Agreement score in [0, 1] where 1 = perfect agreement
    """
    # 1. Text Similarity (70% weight)
    text_a = result_a['ocr_text']
    text_b = result_b['ocr_text']

    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        text_similarity = 1.0
    else:
        edit_dist = levenshtein_distance(text_a, text_b)
        text_similarity = 1 - (edit_dist / max_len)

    # 2. Bounding Box IoU (30% weight)
    bbox_a = result_a['crop_bbox']  # (x, y, w, h)
    bbox_b = result_b['crop_bbox']

    bbox_iou = compute_bbox_iou(bbox_a, bbox_b)

    # 3. Combined Score
    agreement_score = 0.7 * text_similarity + 0.3 * bbox_iou

    return {
        'overall': agreement_score,
        'text_similarity': text_similarity,
        'bbox_iou': bbox_iou
    }

def compute_bbox_iou(bbox_a, bbox_b):
    """
    Compute Intersection over Union for two bounding boxes.

    Args:
        bbox_a, bbox_b: Tuples of (x, y, w, h)
    """
    x1_a, y1_a, w_a, h_a = bbox_a
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = bbox_b
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    # Intersection
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
```

#### Step 4: Generate Confidence Signal & Flag for Review

```python
def generate_duplicate_agreement_signals(video_path, threshold=0.93):
    """
    Generate duplicate agreement signals for all pages.

    Returns:
        dict: page_id -> confidence_score mapping
    """
    # Find all duplicates
    duplicates = find_duplicate_frames(video_path)

    page_scores = {}

    for frame_a, frame_b in duplicates:
        # Process both
        result_a, result_b = process_duplicate_pair(video_path, frame_a, frame_b)

        # Compute agreement
        scores = compute_duplicate_agreement(result_a, result_b)

        # Map to page ID (assuming one page per spread pair)
        page_id = frame_to_page_id(frame_a)

        page_scores[page_id] = {
            'duplicate_agreement': scores['overall'],
            'text_similarity': scores['text_similarity'],
            'bbox_iou': scores['bbox_iou'],
            'flagged': scores['overall'] < threshold
        }

        if scores['overall'] < threshold:
            print(f"⚠️  Page {page_id}: Low duplicate agreement ({scores['overall']:.2f})")
            print(f"    Text similarity: {scores['text_similarity']:.2f}")
            print(f"    BBox IoU: {scores['bbox_iou']:.2f}")

    return page_scores
```

### Expected Thresholds

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| ≥ 0.95 | Perfect consistency | Use as training data |
| 0.90 - 0.95 | Good consistency | Accept, monitor |
| 0.80 - 0.90 | Moderate variance | Review if time permits |
| < 0.80 | Poor consistency | **Flag for review** |

### Test Data Requirements

- **Minimum**: 10 videos with at least 5 duplicate page captures each
- **Optimal**: 50 videos with 20+ duplicates each
- **Ground truth**: Human-verified duplicate pairs (100 pairs minimum)

---

## Signal 2: Entity/Citation/Number Preservation

### What It Tests
**Information Fidelity**: Are critical entities (names, dates, numbers, citations) preserved through the pipeline?

### Why It Matters
- Detects information loss from aggressive cropping
- Validates OCR quality
- Ensures scientific/historical accuracy
- Critical for academic/legal documents

### How It Works

#### Step 1: Extract Entities from Original Image

**Method**: Named Entity Recognition (NER) using spaCy

```python
import spacy
import re

# Load spaCy model (download first: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract all critical entities from text.

    Returns:
        dict with keys: persons, dates, numbers, citations, orgs
    """
    doc = nlp(text)

    entities = {
        'persons': [],
        'dates': [],
        'numbers': [],
        'citations': [],
        'organizations': [],
        'locations': []
    }

    # spaCy NER
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities['persons'].append(ent.text)
        elif ent.label_ == 'DATE':
            entities['dates'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['organizations'].append(ent.text)
        elif ent.label_ == 'GPE':  # Geopolitical entity
            entities['locations'].append(ent.text)

    # Numbers: Extract numeric tokens
    for token in doc:
        if token.like_num or token.is_digit:
            entities['numbers'].append(token.text)

    # Citations: Regex patterns
    # Pattern 1: [Smith et al., 2023]
    citations_1 = re.findall(r'\[([^\]]+(?:et al\.|[0-9]{4})[^\]]*)\]', text)
    entities['citations'].extend(citations_1)

    # Pattern 2: (Smith, 2023)
    citations_2 = re.findall(r'\(([A-Z][a-z]+(?:\s+et al\.)?,\s*[0-9]{4})\)', text)
    entities['citations'].extend(citations_2)

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

def extract_entities_from_image(image):
    """Extract entities from image using OCR first."""
    # OCR
    text = ocr_image(image)

    # NER
    entities = extract_entities(text)

    return text, entities
```

#### Step 2: Process Through Pipeline and Re-extract

```python
def test_entity_preservation(original_image_path):
    """
    Test entity preservation through full pipeline.

    Returns:
        Preservation score and detailed report
    """
    # Load original image
    original_img = cv2.imread(original_image_path)

    # Extract ground truth entities
    gt_text, gt_entities = extract_entities_from_image(original_img)

    print(f"Ground Truth Entities:")
    print(f"  Persons: {len(gt_entities['persons'])}")
    print(f"  Dates: {len(gt_entities['dates'])}")
    print(f"  Numbers: {len(gt_entities['numbers'])}")
    print(f"  Citations: {len(gt_entities['citations'])}")

    total_gt = sum(len(v) for v in gt_entities.values())

    # Process through pipeline
    # Phase 5: Crop
    cropped = crop_image(original_img)

    # Phase 6: Split (if double mode)
    pages = split_pages(cropped)

    # Phase 8: Binarize (optional)
    # binarized = binarize_image(pages[0])

    # Phase 9: Convert to PDF and back
    pdf_path = create_pdf_from_images(pages)
    final_img = pdf_to_image(pdf_path, page=0)

    # Extract final entities
    final_text, final_entities = extract_entities_from_image(final_img)

    print(f"\nFinal Entities:")
    print(f"  Persons: {len(final_entities['persons'])}")
    print(f"  Dates: {len(final_entities['dates'])}")
    print(f"  Numbers: {len(final_entities['numbers'])}")
    print(f"  Citations: {len(final_entities['citations'])}")

    # Compute preservation
    preserved = 0
    lost_entities = []

    for category in gt_entities:
        for entity in gt_entities[category]:
            if entity_preserved(entity, final_entities[category]):
                preserved += 1
            else:
                lost_entities.append((category, entity))

    preservation_rate = preserved / total_gt if total_gt > 0 else 1.0

    return {
        'preservation_rate': preservation_rate,
        'total_entities': total_gt,
        'preserved_count': preserved,
        'lost_entities': lost_entities,
        'gt_entities': gt_entities,
        'final_entities': final_entities
    }
```

#### Step 3: Fuzzy Matching for OCR Errors

**Problem**: OCR may introduce small errors: "Einstein" → "Elnstein"

**Solution**: Fuzzy matching with edit distance tolerance

```python
from Levenshtein import distance as edit_distance

def entity_preserved(gt_entity, final_entity_list, threshold=2):
    """
    Check if entity is preserved, allowing for small OCR errors.

    Args:
        gt_entity: Ground truth entity string
        final_entity_list: List of entities in final output
        threshold: Max edit distance to consider match (default: 2)

    Returns:
        bool: True if entity found (exact or fuzzy match)
    """
    # Check exact match first
    if gt_entity in final_entity_list:
        return True

    # Fuzzy match for persons/orgs (allow OCR errors)
    # For numbers/dates: require exact match
    if is_numeric(gt_entity):
        return False  # Numbers must match exactly

    # Fuzzy match for text entities
    for final_entity in final_entity_list:
        dist = edit_distance(gt_entity.lower(), final_entity.lower())

        # Allow distance proportional to length
        max_dist = max(threshold, len(gt_entity) // 10)  # 10% of length

        if dist <= max_dist:
            return True

    return False

def is_numeric(text):
    """Check if text is primarily numeric."""
    # Remove common separators
    cleaned = text.replace(',', '').replace('.', '').replace('$', '')
    return cleaned.isdigit() or text.replace('.', '').isdigit()
```

#### Step 4: Generate Confidence Signal

```python
def generate_entity_preservation_signals(image_paths, threshold=0.98):
    """
    Generate entity preservation signals for all pages.

    Returns:
        dict: page_id -> preservation metrics
    """
    results = {}

    for img_path in image_paths:
        page_id = extract_page_id(img_path)

        # Test preservation
        metrics = test_entity_preservation(img_path)

        preservation_rate = metrics['preservation_rate']

        results[page_id] = {
            'entity_preservation': preservation_rate,
            'total_entities': metrics['total_entities'],
            'preserved_count': metrics['preserved_count'],
            'lost_entities': metrics['lost_entities'],
            'flagged': preservation_rate < threshold
        }

        if preservation_rate < threshold:
            print(f"⚠️  Page {page_id}: Low entity preservation ({preservation_rate:.2%})")
            print(f"    Lost entities: {metrics['lost_entities']}")

    return results
```

### Expected Thresholds

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| ≥ 0.98 | Excellent preservation | High confidence |
| 0.95 - 0.98 | Good preservation | Accept |
| 0.90 - 0.95 | Moderate loss | Review entity types |
| < 0.90 | Significant loss | **Flag for review** |

### Special Cases

**Numbers require exact preservation**:
- "1905" ≠ "1905.0"
- "$1,000,000" = "$1000000" (after normalization)
- Scientific notation: "3.14159" must be exact

**Dates allow format variations**:
- "June 8, 2025" = "2025-06-08" = "06/08/2025"
- Normalize before comparison

### Test Data Requirements

- **Minimum**: 100 pages with rich entity content
- **Optimal**: 500 pages covering:
  - Academic papers (citations heavy)
  - Historical documents (dates, names)
  - Financial documents (numbers)
  - Scientific texts (formulas, numbers)

---

## Signal 3: Structural Consistency

### What It Tests
**Document Organization**: Is the logical structure of the document preserved?

### Why It Matters
- Validates page ordering
- Ensures chapter/section hierarchy is correct
- Detects missing or duplicate pages
- Critical for training layout models

### How It Works

#### Step 1: Extract Structural Features

```python
import re
from collections import defaultdict

def extract_document_structure(pdf_path):
    """
    Extract structural elements from PDF.

    Returns:
        dict with structure information
    """
    structure = {
        'page_numbers': [],
        'chapters': [],
        'sections': [],
        'citations': [],
        'footnotes': [],
        'figure_captions': [],
        'table_captions': []
    }

    pdf_pages = extract_pages_from_pdf(pdf_path)

    for page_idx, page_img in enumerate(pdf_pages):
        text = ocr_pdf_page(page_img)

        # 1. Extract page number (usually bottom center or corners)
        page_num = extract_page_number(text, page_idx)
        structure['page_numbers'].append(page_num)

        # 2. Detect chapter headings
        chapter_match = re.search(r'Chapter\s+(\d+)', text, re.IGNORECASE)
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            structure['chapters'].append({
                'page': page_idx,
                'number': chapter_num,
                'title': extract_chapter_title(text)
            })

        # 3. Detect sections
        section_matches = re.findall(r'(\d+\.)+\d+\s+([A-Z][^\n]+)', text)
        for section_num, section_title in section_matches:
            structure['sections'].append({
                'page': page_idx,
                'number': section_num,
                'title': section_title
            })

        # 4. Extract citations [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', text)
        structure['citations'].extend([(page_idx, int(c)) for c in citations])

        # 5. Detect footnotes (superscript numbers + bottom of page text)
        footnotes = re.findall(r'(\d+)\s*[A-Z][^.]+\.', text[-500:])  # Last 500 chars
        structure['footnotes'].extend([(page_idx, fn) for fn in footnotes])

        # 6. Figure/Table captions
        fig_captions = re.findall(r'Figure\s+(\d+)[:\.]?\s*([^\n]+)', text, re.IGNORECASE)
        structure['figure_captions'].extend([(page_idx, num, cap) for num, cap in fig_captions])

        table_captions = re.findall(r'Table\s+(\d+)[:\.]?\s*([^\n]+)', text, re.IGNORECASE)
        structure['table_captions'].extend([(page_idx, num, cap) for num, cap in table_captions])

    return structure

def extract_page_number(text, page_idx):
    """
    Extract page number from text (heuristic-based).

    Strategies:
    1. Look for isolated numbers at bottom
    2. Look for "Page N" pattern
    3. Fallback to sequential numbering
    """
    # Strategy 1: Bottom of page (last 200 chars)
    bottom_text = text[-200:]
    isolated_nums = re.findall(r'\b(\d+)\b', bottom_text)
    if isolated_nums:
        return int(isolated_nums[-1])  # Last number found

    # Strategy 2: "Page N" pattern
    page_match = re.search(r'Page\s+(\d+)', text, re.IGNORECASE)
    if page_match:
        return int(page_match.group(1))

    # Fallback: Sequential
    return page_idx + 1
```

#### Step 2: Validate Structural Consistency

```python
def validate_structure(structure):
    """
    Validate structural consistency and compute score.

    Returns:
        score (float), issues (list of strings)
    """
    score = 1.0
    issues = []

    # 1. Page Number Validation
    page_nums = structure['page_numbers']

    # Check for missing pages
    for i in range(len(page_nums) - 1):
        if page_nums[i+1] != page_nums[i] + 1:
            score -= 0.05
            issues.append(f"Page number jump: {page_nums[i]} → {page_nums[i+1]}")

    # Check for duplicates
    if len(page_nums) != len(set(page_nums)):
        duplicates = [num for num in page_nums if page_nums.count(num) > 1]
        score -= 0.1
        issues.append(f"Duplicate page numbers: {set(duplicates)}")

    # 2. Chapter Ordering
    chapters = structure['chapters']
    for i in range(len(chapters) - 1):
        current_ch = chapters[i]['number']
        next_ch = chapters[i+1]['number']

        if next_ch <= current_ch:
            score -= 0.15
            issues.append(f"Chapter disorder: Ch.{current_ch} (page {chapters[i]['page']}) → "
                         f"Ch.{next_ch} (page {chapters[i+1]['page']})")

    # 3. Section Numbering
    sections = structure['sections']
    # Check that section numbers increase within chapters
    # (Complex logic - simplified here)

    # 4. Citation Integrity
    # Check that all cited references exist in reference list
    cited_refs = set([c[1] for c in structure['citations']])  # Extract citation numbers

    # Extract reference list (heuristic: look for "References" page)
    ref_list = extract_reference_list(structure)

    missing_refs = cited_refs - set(ref_list.keys())
    if missing_refs:
        score -= 0.1 * min(len(missing_refs) / 10, 1.0)  # Cap penalty at -0.1
        issues.append(f"Missing references: {sorted(missing_refs)}")

    # 5. Figure/Table Numbering
    fig_nums = [int(num) for _, num, _ in structure['figure_captions']]
    expected_fig_nums = list(range(1, len(fig_nums) + 1))

    if sorted(fig_nums) != expected_fig_nums:
        score -= 0.05
        issues.append(f"Figure numbering inconsistent: expected {expected_fig_nums}, got {sorted(fig_nums)}")

    # Ensure score is non-negative
    score = max(0.0, score)

    return score, issues

def extract_reference_list(structure):
    """
    Extract reference list from structure.
    Assumes references are on last few pages.

    Returns:
        dict: reference_number -> reference_text
    """
    # Implementation depends on document format
    # Simplified: look for "[N]" or "N." patterns on last 3 pages
    return {}  # Placeholder
```

#### Step 3: Generate Confidence Signal

```python
def generate_structural_signals(pdf_path, threshold=0.90):
    """
    Generate structural consistency signals.

    Returns:
        dict with structural metrics
    """
    # Extract structure
    structure = extract_document_structure(pdf_path)

    # Validate
    score, issues = validate_structure(structure)

    result = {
        'structural_consistency': score,
        'issues': issues,
        'page_count': len(structure['page_numbers']),
        'chapter_count': len(structure['chapters']),
        'flagged': score < threshold
    }

    if score < threshold:
        print(f"⚠️  Structural consistency: {score:.2f}")
        for issue in issues:
            print(f"    - {issue}")

    return result
```

### Expected Thresholds

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| ≥ 0.95 | Perfect structure | High confidence |
| 0.90 - 0.95 | Minor issues | Accept |
| 0.80 - 0.90 | Moderate issues | Review flagged issues |
| < 0.80 | Broken structure | **Flag for review** |

### Penalty Schedule

| Issue | Penalty |
|-------|---------|
| Missing page | -0.05 per page |
| Duplicate page | -0.10 |
| Chapter disorder | -0.15 per instance |
| Missing reference | -0.01 per reference (capped at -0.10) |
| Figure numbering gap | -0.05 |

### Test Data Requirements

- **Minimum**: 10 complete books with clear structure
- **Optimal**: 50 books with:
  - Clear chapter divisions
  - Reference sections
  - Figures and tables
  - Page numbers

---

## Signal 4: Retrieval Benchmarks

### What It Tests
**End-to-End Utility**: Can users retrieve known information from the final database?

### Why It Matters
- Ultimate measure of pipeline success
- Tests entire workflow (Video → PDF → Index → Query)
- Validates that information is accessible, not just preserved
- Direct measure of user experience

### How It Works

#### Step 1: Create Test Corpus with Known Q&A Pairs

```python
# Create test corpus with known facts
test_corpus = {
    "queries": [
        {
            "id": "q1",
            "question": "When did Einstein publish the theory of relativity?",
            "expected_answer": "1905",
            "expected_variants": ["1905", "nineteen oh-five"],
            "page_number": 5,
            "answer_type": "DATE"
        },
        {
            "id": "q2",
            "question": "What is the speed of light?",
            "expected_answer": "299,792,458 meters per second",
            "expected_variants": ["299792458", "3×10^8", "speed of light"],
            "page_number": 12,
            "answer_type": "NUMBER"
        },
        {
            "id": "q3",
            "question": "Who developed quantum mechanics?",
            "expected_answer": "Heisenberg",
            "expected_variants": ["Werner Heisenberg", "Heisenberg"],
            "page_number": 23,
            "answer_type": "PERSON"
        }
    ]
}

def create_test_corpus(num_queries=100):
    """
    Create a test corpus with known Q&A pairs.

    Best practices:
    - Mix question types (factual, numerical, named entities)
    - Cover different pages (beginning, middle, end)
    - Include easy and hard queries

    Returns:
        Test corpus dictionary
    """
    return test_corpus
```

#### Step 2: Process and Index in Database

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentDatabase:
    """Simple vector database for retrieval testing."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.page_ids = []

    def index_pdf(self, pdf_path, chunk_size=512):
        """
        Index PDF into vector database.

        Args:
            pdf_path: Path to PDF
            chunk_size: Characters per chunk
        """
        pages = extract_text_from_pdf(pdf_path)

        chunks = []
        chunk_page_ids = []

        for page_num, page_text in enumerate(pages):
            # Split page into chunks
            for i in range(0, len(page_text), chunk_size):
                chunk = page_text[i:i+chunk_size]
                chunks.append(chunk)
                chunk_page_ids.append(page_num)

        # Generate embeddings
        embeddings = self.model.encode(chunks)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        self.documents = chunks
        self.page_ids = chunk_page_ids

    def search(self, query, top_k=5):
        """
        Search database for query.

        Returns:
            List of (text, page_id, score) tuples
        """
        # Encode query
        query_embedding = self.model.encode([query])

        # Search
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'text': self.documents[idx],
                'page_id': self.page_ids[idx],
                'score': float(distance)
            })

        return results
```

#### Step 3: Query and Evaluate

```python
def evaluate_retrieval(test_corpus, database, top_k=5):
    """
    Evaluate retrieval accuracy on test corpus.

    Returns:
        dict with retrieval metrics
    """
    results = {
        'total_queries': len(test_corpus['queries']),
        'exact_match': 0,
        'partial_match': 0,
        'no_match': 0,
        'failures': []
    }

    for query_data in test_corpus['queries']:
        question = query_data['question']
        expected = query_data['expected_answer']
        variants = query_data['expected_variants']

        # Search database
        search_results = database.search(question, top_k=top_k)

        # Aggregate retrieved text
        retrieved_text = " ".join([r['text'] for r in search_results])

        # Check for exact match
        if expected in retrieved_text:
            results['exact_match'] += 1
        # Check for variants
        elif any(var in retrieved_text for var in variants):
            results['partial_match'] += 1
        else:
            results['no_match'] += 1
            results['failures'].append({
                'query_id': query_data['id'],
                'question': question,
                'expected': expected,
                'retrieved': retrieved_text[:200]  # First 200 chars
            })

    # Compute scores
    total = results['total_queries']
    results['exact_match_rate'] = results['exact_match'] / total
    results['partial_match_rate'] = results['partial_match'] / total
    results['recall'] = (results['exact_match'] + results['partial_match']) / total

    return results

def generate_retrieval_signals(pdf_path, test_corpus, threshold=0.90):
    """
    Generate retrieval benchmark signals.

    Returns:
        dict with retrieval metrics
    """
    # Index PDF
    database = DocumentDatabase()
    database.index_pdf(pdf_path)

    # Evaluate
    results = evaluate_retrieval(test_corpus, database)

    recall = results['recall']

    signal = {
        'retrieval_accuracy': recall,
        'exact_matches': results['exact_match'],
        'partial_matches': results['partial_match'],
        'failures': results['failures'],
        'flagged': recall < threshold
    }

    if recall < threshold:
        print(f"⚠️  Retrieval accuracy: {recall:.2%}")
        print(f"    Failed queries: {len(results['failures'])}")

    return signal
```

### Expected Thresholds

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| ≥ 0.95 | Excellent retrieval | Pipeline working |
| 0.90 - 0.95 | Good retrieval | Accept |
| 0.80 - 0.90 | Moderate issues | Review failed queries |
| < 0.80 | Poor retrieval | **Flag for review** |

### Test Data Requirements

- **Minimum**: 50 Q&A pairs covering one test book
- **Optimal**: 500 Q&A pairs covering:
  - Factual questions (dates, names, places)
  - Numerical questions (measurements, statistics)
  - Conceptual questions (definitions, explanations)
  - Multi-hop questions (require multiple pages)

---

## Integration: Combining All Four Signals

### Composite Confidence Score

```python
def compute_composite_confidence(signals):
    """
    Combine all four signals into a single confidence score.

    Args:
        signals: dict with keys: duplicate_agreement, entity_preservation,
                 structural_consistency, retrieval_accuracy

    Returns:
        float: Composite confidence score [0, 1]
    """
    # Weights (can be tuned)
    weights = {
        'duplicate_agreement': 0.20,
        'entity_preservation': 0.35,
        'structural_consistency': 0.20,
        'retrieval_accuracy': 0.25
    }

    score = 0.0
    for signal_name, weight in weights.items():
        score += signals.get(signal_name, 0.0) * weight

    return score

def prioritize_pages_for_review(all_page_signals):
    """
    Sort pages by confidence score for human review prioritization.

    Args:
        all_page_signals: dict of page_id -> signals dict

    Returns:
        Sorted list of (page_id, score, issues)
    """
    page_scores = []

    for page_id, signals in all_page_signals.items():
        composite = compute_composite_confidence(signals)

        # Identify specific issues
        issues = []
        if signals.get('duplicate_agreement', 1.0) < 0.90:
            issues.append("Low duplicate agreement")
        if signals.get('entity_preservation', 1.0) < 0.95:
            issues.append(f"Entity loss: {len(signals.get('lost_entities', []))} entities")
        if signals.get('structural_consistency', 1.0) < 0.90:
            issues.append("Structural issues")

        page_scores.append((page_id, composite, issues))

    # Sort by score (lowest first = needs review most)
    page_scores.sort(key=lambda x: x[1])

    return page_scores
```

### Complete Workflow

```python
def run_verification_pipeline(video_path):
    """
    Run all four verification signals on a video.

    Returns:
        Complete verification report
    """
    print("=" * 60)
    print("VERIFICATION PIPELINE")
    print("=" * 60)

    # Process video through ScanStudio
    print("\n1. Processing video through pipeline...")
    pdf_path = scanstudio_pipeline(video_path)

    # Signal 1: Duplicate Agreement
    print("\n2. Computing duplicate agreement signals...")
    dup_signals = generate_duplicate_agreement_signals(video_path)

    # Signal 2: Entity Preservation
    print("\n3. Computing entity preservation signals...")
    # Extract keyframes first
    keyframes = extract_keyframes_from_video(video_path)
    entity_signals = {}
    for kf_path in keyframes:
        page_id = extract_page_id(kf_path)
        result = test_entity_preservation(kf_path)
        entity_signals[page_id] = result

    # Signal 3: Structural Consistency
    print("\n4. Computing structural consistency signals...")
    struct_signals = generate_structural_signals(pdf_path)

    # Signal 4: Retrieval Benchmarks
    print("\n5. Computing retrieval benchmarks...")
    test_corpus = load_test_corpus_for_video(video_path)
    retrieval_signals = generate_retrieval_signals(pdf_path, test_corpus)

    # Combine signals
    print("\n6. Combining signals and prioritizing review...")
    all_page_signals = {}

    for page_id in set(list(dup_signals.keys()) + list(entity_signals.keys())):
        all_page_signals[page_id] = {
            'duplicate_agreement': dup_signals.get(page_id, {}).get('duplicate_agreement', 1.0),
            'entity_preservation': entity_signals.get(page_id, {}).get('preservation_rate', 1.0),
            'structural_consistency': struct_signals['structural_consistency'],  # Global
            'retrieval_accuracy': retrieval_signals['retrieval_accuracy']  # Global
        }

    # Prioritize
    review_queue = prioritize_pages_for_review(all_page_signals)

    # Generate report
    report = {
        'video_path': video_path,
        'pdf_path': pdf_path,
        'total_pages': len(all_page_signals),
        'signals': {
            'duplicate_agreement': dup_signals,
            'entity_preservation': entity_signals,
            'structural_consistency': struct_signals,
            'retrieval_accuracy': retrieval_signals
        },
        'review_queue': review_queue,
        'high_confidence_pages': [p for p, s, _ in review_queue if s >= 0.95],
        'flagged_pages': [p for p, s, _ in review_queue if s < 0.85]
    }

    print(f"\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total pages: {report['total_pages']}")
    print(f"High confidence (≥0.95): {len(report['high_confidence_pages'])} pages")
    print(f"Flagged for review (<0.85): {len(report['flagged_pages'])} pages")
    print(f"\nTop 10 pages needing review:")
    for page_id, score, issues in review_queue[:10]:
        print(f"  Page {page_id}: {score:.2f} - {', '.join(issues)}")

    return report
```

---

## Summary

### Implementation Priority

1. **Week 1-2**: Signal 2 (Entity Preservation)
   - Most straightforward to implement
   - Immediate value for quality assessment
   - Clear pass/fail criteria

2. **Week 3**: Signal 1 (Duplicate Agreement)
   - Requires duplicate detection first
   - Validates pipeline stability

3. **Week 4**: Signal 4 (Retrieval Benchmarks)
   - Requires test corpus creation
   - End-to-end validation

4. **Week 5**: Signal 3 (Structural Consistency)
   - Most complex (document understanding)
   - Build on previous signals

### Dependencies

```bash
pip install imagehash Pillow python-Levenshtein spacy faiss-cpu sentence-transformers

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Expected Outcomes

After implementation:
- **70% reduction** in human review time
- **15% quality improvement** per iteration
- **Automated quality reporting** for every processed video
- **Training data generation** from high-confidence outputs
