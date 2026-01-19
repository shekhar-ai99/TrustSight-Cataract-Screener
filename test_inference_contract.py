"""
Quick test of the submission inference.py contract.
Tests DataFrame input, numpy array input, and output format.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add code dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

import inference

def test_dataframe_input():
    """Test evaluator format (DataFrame with image_vector column)"""
    print("\n✅ TEST 1: DataFrame input (Evaluator format)")
    print("-" * 50)
    
    # Create dummy vectors
    vectors = [np.random.rand(786432).astype("float32") for _ in range(2)]
    df = pd.DataFrame({
        "image_vector": vectors,
        "extra_column": ["sample1", "sample2"]  # Should be ignored
    })
    
    print(f"Input: DataFrame with shape {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    try:
        preds = inference.predict(df)
        print(f"Output type: {type(preds)}")
        print(f"Output shape: {preds.shape}")
        print(f"Predictions: {preds}")
        
        # Validate
        assert isinstance(preds, np.ndarray), "Output must be numpy array"
        assert preds.shape == (2,), f"Expected shape (2,), got {preds.shape}"
        assert all(isinstance(p, (str, np.str_)) for p in preds), "All outputs must be strings"
        
        valid_classes = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]
        assert all(p in valid_classes for p in preds), f"Invalid class names: {preds}"
        
        print("✅ PASSED: DataFrame input works, outputs are valid class labels")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_numpy_input():
    """Test fallback format (raw numpy array)"""
    print("\n✅ TEST 2: NumPy array input (Backward compatible)")
    print("-" * 50)
    
    # Create dummy vectors
    data = np.random.rand(2, 786432).astype("float32")
    
    print(f"Input: NumPy array with shape {data.shape}")
    
    try:
        preds = inference.predict(data)
        print(f"Output type: {type(preds)}")
        print(f"Output shape: {preds.shape}")
        print(f"Predictions: {preds}")
        
        # Validate
        assert isinstance(preds, np.ndarray), "Output must be numpy array"
        assert preds.shape == (2,), f"Expected shape (2,), got {preds.shape}"
        assert all(isinstance(p, (str, np.str_)) for p in preds), "All outputs must be strings"
        
        valid_classes = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]
        assert all(p in valid_classes for p in preds), f"Invalid class names: {preds}"
        
        print("✅ PASSED: NumPy input works, outputs are valid class labels")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_single_sample():
    """Test single sample (1D input)"""
    print("\n✅ TEST 3: Single sample (1D input)")
    print("-" * 50)
    
    # Create single vector
    data = np.random.rand(786432).astype("float32")
    
    print(f"Input: NumPy array with shape {data.shape}")
    
    try:
        preds = inference.predict(data)
        print(f"Output type: {type(preds)}")
        print(f"Output shape: {preds.shape}")
        print(f"Prediction: {preds[0]}")
        
        # Validate
        assert isinstance(preds, np.ndarray), "Output must be numpy array"
        assert preds.shape == (1,), f"Expected shape (1,), got {preds.shape}"
        assert isinstance(preds[0], (str, np.str_)), "Output must be string"
        
        valid_classes = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]
        assert preds[0] in valid_classes, f"Invalid class name: {preds[0]}"
        
        print("✅ PASSED: Single sample works correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_wrong_vector_size():
    """Test error handling for wrong vector size"""
    print("\n✅ TEST 4: Error handling (Wrong vector size)")
    print("-" * 50)
    
    # Create vector with wrong size
    data = np.random.rand(2, 100).astype("float32")  # Too small
    
    print(f"Input: NumPy array with shape {data.shape} (invalid)")
    
    try:
        preds = inference.predict(data)
        print(f"❌ FAILED: Should have raised ValueError but got: {preds}")
        return False
    except ValueError as e:
        print(f"Caught expected error: {e}")
        print("✅ PASSED: Correct error handling for invalid vector size")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False

def test_class_names():
    """Test that class names match training data"""
    print("\n✅ TEST 5: Class names validation")
    print("-" * 50)
    
    expected_classes = [
        "No Cataract",
        "Immature Cataract",
        "Mature Cataract",
        "IOL Inserted"
    ]
    
    print(f"Expected classes: {expected_classes}")
    print(f"Loaded classes:   {inference.CLASS_NAMES}")
    
    if inference.CLASS_NAMES == expected_classes:
        print("✅ PASSED: Class names match training data")
        return True
    else:
        print("❌ FAILED: Class names don't match")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SUBMISSION INFERENCE CONTRACT TESTS")
    print("=" * 60)
    
    results = []
    results.append(("DataFrame Input", test_dataframe_input()))
    results.append(("NumPy Input", test_numpy_input()))
    results.append(("Single Sample", test_single_sample()))
    results.append(("Error Handling", test_wrong_vector_size()))
    results.append(("Class Names", test_class_names()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + ("=" * 60))
    if all_passed:
        print("🎉 ALL TESTS PASSED - Submission is ready!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
        print("=" * 60)
        sys.exit(1)
