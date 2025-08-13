import grain.python as grain
import ml_collections
from etils import epath
from orbax import checkpoint as orbax_checkpoint


def _get_checkpoint_manager(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    create: bool = True,
    is_grain_loader: bool = True,
) -> orbax_checkpoint.CheckpointManager:
    """Loads the orbax checkpoint manager for train state and data iterator."""
    # The keys in this dict should match the keys in `checkpointed_state`.
    checkpointers = dict(
        train_state=orbax_checkpoint.PyTreeCheckpointer(),
    )

    # Only add train_iter checkpointer for grain loaders
    if is_grain_loader:
        checkpointers["train_iter"] = (
            orbax_checkpoint.Checkpointer(  # pytype:disable=unsupported-operands
                grain.PyGrainCheckpointHandler()
            )
        )  # pytype:disable=wrong-arg-types

    checkpoint_dir = epath.Path(workdir) / "checkpoints"
    # keep_period = (
    #     config.checkpoint_keep_period if config.checkpoint_keep_period > 0 else None
    # )
    return orbax_checkpoint.CheckpointManager(
        checkpoint_dir,
        checkpointers=checkpointers,
        options=orbax_checkpoint.CheckpointManagerOptions(
            create=create,
            # preservation_policy=orbax_checkpoint.checkpoint_managers.LatestN(n=20),
            best_fn=lambda x: x["validation_loss"]
            if "validation_loss" in x
            else x["loss"],
            best_mode="min",
            max_to_keep=20,
        ),
    )


def get_checkpoint_managers(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    olddir: epath.PathLike | None = None,
    is_grain_loader: bool = True,
) -> tuple[orbax_checkpoint.CheckpointManager, orbax_checkpoint.CheckpointManager]:
    """Get save and load checkpoint managers.

    Args:
        config: Configuration to use.
        workdir: Working directory for saving checkpoints.
        olddir: Optional directory to load old checkpoints from. If provided,
            load manager will point to olddir, otherwise same as save manager.
        is_grain_loader: Whether the data loader is a grain loader.

    Returns:
        A tuple of (save_manager, load_manager).
    """
    # Create checkpoint manager for saving (new checkpoints)
    save_checkpoint_manager = _get_checkpoint_manager(
        config, workdir, create=True, is_grain_loader=is_grain_loader
    )

    # Create checkpoint manager for loading (old checkpoints if olddir is provided)
    if olddir is not None:
        load_checkpoint_manager = _get_checkpoint_manager(
            config, olddir, create=False, is_grain_loader=is_grain_loader
        )
    else:
        load_checkpoint_manager = save_checkpoint_manager

    return save_checkpoint_manager, load_checkpoint_manager
