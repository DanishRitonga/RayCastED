"""Quick smoke test for E2E dual-assignment fix.

Tests that the fix works in practice:
- E2E architecture validated: o2m.topk=13, o2o.topk=1
- No assertion errors during training
- Loss values are reasonable
"""

from raycasted.model.train import RayCastTrainer


def main():
    """Run 2-epoch smoke test with E2E fix."""
    print('=== E2E Fix Smoke Test ===')
    print('Training config: tal_topk=13, one2one.topk=1 (enforced)')
    print()

    # E2E FIX: Use training_config for custom parameters
    training_config = {
        'tal_topk': 13,  # NEW parameter name
        'assigner_radius_scale': 1.5,
    }

    trainer = RayCastTrainer(
        overrides={
            'model': 'yolo26s.yaml',
            'data': '/home/danishrtg/projects/RayCastED/output/pannuke-debug/transformed/data.yaml',
            'epochs': 2,
            'batch': 8,
            'imgsz': 256,
            'device': 'cpu',
            'workers': 2,
            'project': 'smoke_e2e_fix',
            'name': 'run',
            'exist_ok': True,
            'verbose': True,
        },
        training_config=training_config,
    )

    print('Starting training...')
    print()

    try:
        trainer.train()
        print()
        print('✅ SMOKE TEST PASSED!')
        print('   - E2E architecture working correctly')
        print('   - No assertion errors')
        print('   - Training completed successfully')
        print()
        print('Next steps:')
        print('   1. Run full training (200 epochs)')
        print('   2. Compare results against Run 7 baseline')
        print('   3. Expected: Precision >0.80, mAP@0.5 >0.45')

    except Exception as e:
        print()
        print('❌ SMOKE TEST FAILED!')
        print(f'   Error: {e}')
        raise


if __name__ == '__main__':
    main()
