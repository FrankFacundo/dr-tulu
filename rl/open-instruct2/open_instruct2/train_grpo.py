from __future__ import annotations

from open_instruct2.config import apply_smoke_test_overrides, parse_args
from open_instruct2.trainer import SingleProcessGRPOTrainer


def main() -> None:
    args = apply_smoke_test_overrides(parse_args())
    trainer = SingleProcessGRPOTrainer(args)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
