import ml_collections
from etils import epath
from orbax import checkpoint as orbax_checkpoint


def get_checkpoint_manager(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    create: bool = True,
) -> orbax_checkpoint.CheckpointManager:
    """Create a checkpoint manager with preemption tolerance."""
    # Use checkpoint_dir from config if specified, otherwise default to workdir/checkpoints
    if hasattr(config, 'checkpoint_dir') and config.checkpoint_dir and isinstance(config.checkpoint_dir, str):
        checkpoint_dir = epath.Path(config.checkpoint_dir)
    else:
        checkpoint_dir = epath.Path(workdir) / "checkpoints"
    
    # Ensure checkpoint_every_steps is an integer
    checkpoint_every_steps = config.get("checkpoint_every_steps", 10000)
    if not isinstance(checkpoint_every_steps, int):
        checkpoint_every_steps = 10000
    
    return orbax_checkpoint.CheckpointManager(
        checkpoint_dir,
        options=orbax_checkpoint.CheckpointManagerOptions(
            create=create,
            best_fn=lambda x: x.get("validation_loss", x.get("loss", float('inf'))),
            best_mode="min",
            max_to_keep=20,
            save_interval_steps=checkpoint_every_steps,
        ),
    )


def get_checkpoint_managers(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    olddir: epath.PathLike | None = None,
) -> tuple[orbax_checkpoint.CheckpointManager, orbax_checkpoint.CheckpointManager]:
    """Get save and load checkpoint managers with preemption tolerance.

    Args:
        config: Configuration to use.
        workdir: Working directory for saving checkpoints.
        olddir: Optional directory to load old checkpoints from. If provided,
            load manager will point to olddir, otherwise same as save manager.

    Returns:
        A tuple of (save_manager, load_manager).
    """
    # Create checkpoint manager for saving (new checkpoints)
    save_checkpoint_manager = get_checkpoint_manager(
        config, workdir, create=True
    )

    # Create checkpoint manager for loading (old checkpoints if olddir is provided)
    if olddir is not None:
        load_checkpoint_manager = get_checkpoint_manager(
            config, olddir, create=False
        )
    else:
        load_checkpoint_manager = save_checkpoint_manager

    return save_checkpoint_manager, load_checkpoint_manager
