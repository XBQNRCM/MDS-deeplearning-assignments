"""
TinyImageNet 训练问题快速诊断脚本
直接在notebook中运行此脚本来诊断问题
"""

import torch
import torch.nn as nn
from models_imagenet import get_model
from dataset_utils import get_dataset_loaders
from training_utils import train_model, get_optimizer, get_optimizer_config

def diagnose_training():
    """快速诊断TinyImageNet训练问题"""
    
    print("="*70)
    print("TinyImageNet Training Diagnosis")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # 1. 检查数据加载
    print("\n" + "="*70)
    print("Step 1: Checking data loading...")
    print("="*70)
    
    try:
        train_loader, test_loader, num_classes = get_dataset_loaders(
            'tiny_imagenet', batch_size=64, num_workers=2, augmentation_type='basic'
        )
        print(f"✓ Data loaded successfully")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # 检查一个batch
        data, target = next(iter(train_loader))
        print(f"  - Batch data shape: {data.shape}")
        print(f"  - Target range: [{target.min()}, {target.max()}]")
        
        if num_classes != 200:
            print(f"  ⚠️  WARNING: Expected 200 classes, got {num_classes}")
            return False
            
    except Exception as e:
        print(f"✗ Data loading FAILED: {e}")
        return False
    
    # 2. 测试每个模型
    print("\n" + "="*70)
    print("Step 2: Testing each model (5 epochs)...")
    print("="*70)
    
    models_to_test = ['baseline', 'residual', 'deeper_wider', 'se_attention']
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for model_name in models_to_test:
        print(f"\nTesting: {model_name}")
        print("-" * 70)
        
        try:
            # 创建模型
            model = get_model(model_name, num_classes).to(device)
            print(f"  ✓ Model created ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
            
            # 配置优化器
            optimizer_config = get_optimizer_config('Adam')
            optimizer = get_optimizer(model, 'Adam', **optimizer_config)
            print(f"  ✓ Optimizer configured (lr={optimizer_config['lr']})")
            
            # 快速训练5个epoch
            history = train_model(
                model, train_loader, test_loader,
                num_epochs=5, optimizer=optimizer, criterion=criterion, 
                device=device, print_every=5
            )
            
            # 检查结果
            initial_acc = history['test_acc'][0]
            final_acc = history['test_acc'][-1]
            best_acc = max(history['test_acc'])
            
            results[model_name] = {
                'initial': initial_acc,
                'final': final_acc,
                'best': best_acc,
                'learning': final_acc > initial_acc + 1.0  # 至少提升1%
            }
            
            print(f"  ✓ Training completed")
            print(f"    - Initial accuracy: {initial_acc:.2f}%")
            print(f"    - Final accuracy: {final_acc:.2f}%")
            print(f"    - Improvement: {final_acc - initial_acc:.2f}%")
            
            if not results[model_name]['learning']:
                print(f"    ⚠️  WARNING: Model is NOT learning!")
            else:
                print(f"    ✓ Model is learning normally")
                
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results[model_name] = {'error': str(e)}
    
    # 3. 总结
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    all_learning = all(
        r.get('learning', False) for r in results.values() if 'error' not in r
    )
    
    print(f"\nModel Performance (5 epochs):")
    print("-" * 70)
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:20s}: ERROR - {result['error']}")
        else:
            status = "✓ LEARNING" if result['learning'] else "✗ NOT LEARNING"
            print(f"{model_name:20s}: {result['final']:6.2f}% (Δ{result['final']-result['initial']:+.2f}%) - {status}")
    
    print("\n" + "="*70)
    if all_learning:
        print("✓ DIAGNOSIS: All models are learning normally!")
        print("\nRECOMMENDATION:")
        print("  - Your models are working correctly")
        print("  - TinyImageNet is difficult and needs 30-50 epochs to see good results")
        print("  - Expected accuracy after 50 epochs: 40-55%")
        print("  - Be patient and let the training complete")
    else:
        print("⚠️  DIAGNOSIS: Some models are NOT learning!")
        print("\nPOSSIBLE CAUSES:")
        print("  1. Check if you're using the correct data loader")
        print("  2. Verify num_classes=200 (not 10)")
        print("  3. Make sure you run all cells in order")
        print("  4. Try restarting the kernel and running from the beginning")
        print("\nNEXT STEPS:")
        print("  1. Restart Jupyter kernel")
        print("  2. Run all cells in order without skipping")
        print("  3. Check TINYIMAGENET_DEBUG.md for detailed troubleshooting")
    
    print("="*70)
    
    return all_learning


if __name__ == "__main__":
    diagnose_training()

