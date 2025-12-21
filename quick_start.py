"""
Quick Start Demo - CDS6334 Image Enhancement
This script demonstrates how to use the enhancement and evaluation tools.
"""

import os
from pathlib import Path

def setup_folders():
    """Create necessary folder structure."""
    folders = [
        "dataset/01. Hazy - Raw",
        "dataset/01. Hazy - Enhanced (GT)",
        "dataset/02. Low Light - Raw",
        "dataset/02. Low Light - Enhanced (GT)",
        "output"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print("✓ Folder structure created")


def check_dataset():
    """Check if dataset is properly set up."""
    required_folders = [
        "dataset/01. Hazy - Raw",
        "dataset/01. Hazy - Enhanced (GT)",
        "dataset/02. Low Light - Raw",
        "dataset/02. Low Light - Enhanced (GT)"
    ]
    
    all_exist = all(os.path.exists(f) for f in required_folders)
    
    if all_exist:
        # Count images
        hazy_count = len([f for f in os.listdir("dataset/01. Hazy - Raw") 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        lowlight_count = len([f for f in os.listdir("dataset/02. Low Light - Raw") 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        print(f"\n{'='*60}")
        print("Dataset Status:")
        print(f"{'='*60}")
        print(f"Hazy images found: {hazy_count}")
        print(f"Low-light images found: {lowlight_count}")
        print(f"{'='*60}\n")
        
        if hazy_count > 0 or lowlight_count > 0:
            return True
        else:
            print("⚠ Warning: No images found in dataset folders")
            return False
    else:
        print("⚠ Warning: Dataset folders not properly set up")
        print("\nPlease ensure the following folders exist and contain images:")
        for folder in required_folders:
            status = "✓" if os.path.exists(folder) else "✗"
            print(f"  {status} {folder}")
        return False


def run_enhancement():
    """Run the enhancement pipeline."""
    print("\n" + "="*60)
    print("STEP 1: Running Image Enhancement")
    print("="*60 + "\n")
    
    try:
        from enhancement import ImageEnhancer
        
        enhancer = ImageEnhancer()
        
        # Process hazy images
        if os.path.exists("dataset/01. Hazy - Raw"):
            print("Processing hazy images...")
            hazy_count = enhancer.process_folder(
                "dataset/01. Hazy - Raw",
                "output/hazy-student-enhanced",
                'hazy'
            )
            print(f"✓ Processed {hazy_count} hazy images")
        
        # Process low-light images
        if os.path.exists("dataset/02. Low Light - Raw"):
            print("\nProcessing low-light images...")
            lowlight_count = enhancer.process_folder(
                "dataset/02. Low Light - Raw",
                "output/lowlight-student-enhanced",
                'lowlight'
            )
            print(f"✓ Processed {lowlight_count} low-light images")
        
        print("\n✓ Enhancement complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error during enhancement: {str(e)}")
        return False


def run_evaluation():
    """Run the evaluation pipeline."""
    print("\n" + "="*60)
    print("STEP 2: Running Evaluation")
    print("="*60 + "\n")
    
    try:
        from evaluation import ImageEvaluator
        
        evaluator = ImageEvaluator()
        
        # Evaluate hazy images
        if os.path.exists("output/hazy-student-enhanced"):
            print("Evaluating hazy images...")
            df_hazy = evaluator.evaluate_folder(
                raw_folder="dataset/01. Hazy - Raw",
                gt_folder="dataset/01. Hazy - Enhanced (GT)",
                enhanced_folder="output/hazy-student-enhanced",
                category='hazy',
                visualization_folder="output/visualizations/hazy"
            )
            
            if not df_hazy.empty:
                summary = evaluator.generate_summary_report(df_hazy, 'hazy')
                print(f"\n✓ Hazy Images Summary:")
                print(f"   Avg PSNR: {summary['Avg_PSNR']:.2f} dB")
                print(f"   Avg SSIM: {summary['Avg_SSIM']:.4f}")
        
        # Evaluate low-light images
        if os.path.exists("output/lowlight-student-enhanced"):
            print("\nEvaluating low-light images...")
            df_lowlight = evaluator.evaluate_folder(
                raw_folder="dataset/02. Low Light - Raw",
                gt_folder="dataset/02. Low Light - Enhanced (GT)",
                enhanced_folder="output/lowlight-student-enhanced",
                category='lowlight',
                visualization_folder="output/visualizations/lowlight"
            )
            
            if not df_lowlight.empty:
                summary = evaluator.generate_summary_report(df_lowlight, 'lowlight')
                print(f"\n✓ Low-Light Images Summary:")
                print(f"   Avg PSNR: {summary['Avg_PSNR']:.2f} dB")
                print(f"   Avg SSIM: {summary['Avg_SSIM']:.4f}")
        
        # Save results
        evaluator.save_results_to_excel("output/evaluation_results.xlsx")
        
        print("\n✓ Evaluation complete!")
        print("✓ Results saved to: output/evaluation_results.xlsx")
        return True
        
    except Exception as e:
        print(f"✗ Error during evaluation: {str(e)}")
        return False


def generate_sample_reports():
    """Generate sample visual reports."""
    print("\n" + "="*60)
    print("STEP 3: Generating Sample Reports (Optional)")
    print("="*60 + "\n")
    
    try:
        from utils import batch_create_reports
        
        # Hazy reports
        if os.path.exists("output/hazy-student-enhanced"):
            print("Creating hazy image reports...")
            batch_create_reports(
                raw_folder="dataset/01. Hazy - Raw",
                gt_folder="dataset/01. Hazy - Enhanced (GT)",
                enhanced_folder="output/hazy-student-enhanced",
                report_folder="output/reports/hazy",
                num_samples=5
            )
        
        # Low-light reports
        if os.path.exists("output/lowlight-student-enhanced"):
            print("\nCreating low-light image reports...")
            batch_create_reports(
                raw_folder="dataset/02. Low Light - Raw",
                gt_folder="dataset/02. Low Light - Enhanced (GT)",
                enhanced_folder="output/lowlight-student-enhanced",
                report_folder="output/reports/lowlight",
                num_samples=5
            )
        
        print("\n✓ Sample reports generated in output/reports/")
        return True
        
    except Exception as e:
        print(f"⚠ Could not generate reports: {str(e)}")
        return False


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Review your enhanced images in the output/ folder

2. Check the evaluation results in output/evaluation_results.xlsx

3. Look at the visualizations in output/visualizations/

4. If needed, adjust parameters in enhancement.py:
   - For hazy images: self.hazy_params
   - For low-light images: self.lowlight_params

5. Re-run the pipeline to improve results:
   python quick_start.py

6. Prepare your report with:
   - Sample before/after images
   - Histograms
   - Metrics tables
   - Analysis and discussion

7. Create submission ZIP with:
   - enhancement.py
   - evaluation.py
   - utils.py
   - requirements.txt
   - output/ folder
   - Your report (PDF)
""")
    print("="*60 + "\n")


def main():
    """Main quick start function."""
    print("\n" + "="*60)
    print("CDS6334 Image Enhancement - Quick Start")
    print("="*60 + "\n")
    
    # Setup folders
    setup_folders()
    
    # Check dataset
    dataset_ready = check_dataset()
    
    if not dataset_ready:
        print("\n⚠ Please place your dataset in the correct folders before proceeding.")
        print("\nExpected structure:")
        print("  dataset/")
        print("  ├── 01. Hazy - Raw/")
        print("  ├── 01. Hazy - Enhanced (GT)/")
        print("  ├── 02. Low Light - Raw/")
        print("  └── 02. Low Light - Enhanced (GT)/")
        return
    
    # Ask user if they want to proceed
    print("Ready to process images!")
    response = input("\nProceed with enhancement? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Cancelled. Run this script again when ready.")
        return
    
    # Run pipeline
    success_enhance = run_enhancement()
    
    if success_enhance:
        success_eval = run_evaluation()
        
        if success_eval:
            # Ask about sample reports
            response = input("\nGenerate sample visual reports? (y/n): ").lower().strip()
            if response == 'y':
                generate_sample_reports()
    
    # Print next steps
    print_next_steps()
    
    print("✓ Quick start complete!")


if __name__ == "__main__":
    main()
