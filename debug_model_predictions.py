import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

# Load the model
model = tf.keras.models.load_model('trained_model.keras')
class_names = ['Black Tan Spot', 'Healthy Wheat', 'Yellow Rust', 'brown rust']

print("🔍 DEBUGGING MODEL PREDICTIONS")
print("=" * 50)

def test_specific_images():
    """Test specific images from each class to see confusion patterns"""
    
    # Test directories
    test_dirs = {
        'Yellow Rust': 'val/Yellow Rust',  # Adjust path as needed
        'brown rust': 'val/brown rust'     # Adjust path as needed
    }
    
    results = {}
    
    for true_class, directory in test_dirs.items():
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            continue
            
        print(f"\n🧪 Testing {true_class} images...")
        
        # Get first 5 images from this class
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        
        class_predictions = []
        
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            
            try:
                # Load and preprocess exactly like your backend
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image = image.resize((128, 128), Image.Resampling.LANCZOS)
                img_array = np.array(image, dtype=np.float32) / 255.0
                input_arr = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = model.predict(input_arr, verbose=0)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0]) * 100
                
                class_predictions.append({
                    'file': img_file,
                    'predicted': predicted_class,
                    'confidence': confidence,
                    'all_scores': {class_names[i]: predictions[0][i]*100 for i in range(len(class_names))}
                })
                
                print(f"  📄 {img_file}")
                print(f"     Predicted: {predicted_class} ({confidence:.1f}%)")
                print(f"     Yellow Rust: {predictions[0][2]*100:.1f}% | Brown Rust: {predictions[0][3]*100:.1f}%")
                
            except Exception as e:
                print(f"  ❌ Error with {img_file}: {e}")
        
        results[true_class] = class_predictions
    
    return results

def analyze_confusion():
    """Analyze which classes are being confused"""
    results = test_specific_images()
    
    print(f"\n📊 CONFUSION ANALYSIS")
    print("=" * 30)
    
    for true_class, predictions in results.items():
        if not predictions:
            continue
            
        correct = sum(1 for p in predictions if p['predicted'] == true_class)
        total = len(predictions)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"\n{true_class}:")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        # Show what it's being confused with
        confusion_count = {}
        for p in predictions:
            pred_class = p['predicted']
            confusion_count[pred_class] = confusion_count.get(pred_class, 0) + 1
        
        print("  Predicted as:")
        for pred_class, count in confusion_count.items():
            percentage = (count / total) * 100
            print(f"    {pred_class}: {count}/{total} ({percentage:.1f}%)")

def visualize_feature_differences():
    """Create a visualization to see the differences between Yellow and Brown rust"""
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Yellow Rust vs Brown Rust - Sample Images', fontsize=16)
    
    classes_to_compare = ['Yellow Rust', 'brown rust']
    
    for class_idx, class_name in enumerate(classes_to_compare):
        directory = f'val/{class_name}'  # Adjust path as needed
        
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            continue
        
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        
        for img_idx, img_file in enumerate(image_files):
            if img_idx >= 5:
                break
                
            img_path = os.path.join(directory, img_file)
            
            try:
                # Load image
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize for display
                display_image = image.resize((128, 128), Image.Resampling.LANCZOS)
                
                # Predict
                img_array = np.array(display_image, dtype=np.float32) / 255.0
                input_arr = np.expand_dims(img_array, axis=0)
                predictions = model.predict(input_arr, verbose=0)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0]) * 100
                
                # Plot
                ax = axes[class_idx, img_idx]
                ax.imshow(display_image)
                ax.set_title(f'True: {class_name}\nPred: {predicted_class}\n({confidence:.1f}%)', fontsize=8)
                ax.axis('off')
                
                # Color border based on correctness
                if predicted_class == class_name:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)
                else:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                        
            except Exception as e:
                ax = axes[class_idx, img_idx]
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('rust_confusion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Visualization saved as 'rust_confusion_analysis.png'")

if __name__ == "__main__":
    # Run the analysis
    analyze_confusion()
    visualize_feature_differences()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 30)
    print("1. If Yellow Rust accuracy is low, you may need:")
    print("   - More Yellow Rust training images")
    print("   - Better quality/diverse Yellow Rust images")
    print("   - Different data augmentation for rust classes")
    print("2. Consider retraining with class weights")
    print("3. Check if images are correctly labeled")
