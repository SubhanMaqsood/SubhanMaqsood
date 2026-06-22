"""Unit tests for evaluation metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from main import rouge_l_score, lexical_diversity, readability_score, coherence_score

def test_rouge_l_identical():
    assert rouge_l_score("hello world", "hello world") == 1.0

def test_rouge_l_no_overlap():
    score = rouge_l_score("cat sat mat", "dog ran fast")
    assert score == 0.0

def test_lexical_diversity_all_unique():
    score = lexical_diversity("apple banana cherry date")
    assert score == 1.0

def test_lexical_diversity_all_same():
    score = lexical_diversity("the the the the")
    assert score == 0.25

def test_readability_returns_valid_range():
    text = "This is a simple test sentence. It should be easy to read."
    score = readability_score(text)
    assert 0 <= score <= 100

def test_coherence_structured_text():
    structured = "Point 1.\n- Item A\n- Item B\nConclusion."
    plain      = "just a single plain sentence"
    assert coherence_score(structured) >= coherence_score(plain)

if __name__ == "__main__":
    tests = [test_rouge_l_identical, test_rouge_l_no_overlap,
             test_lexical_diversity_all_unique, test_lexical_diversity_all_same,
             test_readability_returns_valid_range, test_coherence_structured_text]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  PASS {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
