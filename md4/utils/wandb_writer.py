"""WandB metric writer implementation."""

from typing import Any, Mapping, Optional
from clu.metric_writers import interface
import wandb

Array = interface.Array
Scalar = interface.Scalar


class WandBWriter(interface.MetricWriter):
    """A metric writer that logs to Weights & Biases (wandb)."""

    def __init__(self, project: Optional[str] = None, **wandb_init_kwargs):
        """Initialize the WandB writer.
        
        Args:
            project: WandB project name. If None, will use wandb defaults.
            **wandb_init_kwargs: Additional arguments to pass to wandb.init()
        """
        self._project = project
        self._wandb_init_kwargs = wandb_init_kwargs
        
        # Initialize wandb immediately
        init_kwargs = self._wandb_init_kwargs.copy()
        if self._project is not None:
            init_kwargs['project'] = self._project
        wandb.init(**init_kwargs)
        self._initialized = True

    def write_summaries(
        self, 
        step: int,
        values: Mapping[str, Array],
        metadata: Optional[Mapping[str, Any]] = None
    ):
        """Saves an arbitrary tensor summary.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the scalar values occurred.
            values: Mapping from tensor keys to tensors.
            metadata: Optional SummaryMetadata.
        """
        raise NotImplementedError("write_summaries is not implemented for WandBWriter yet.")

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar values for the step.

        Args:
            step: Step at which the scalar values occurred.
            scalars: Mapping from metric name to value.
        """        
        # Convert scalars to Python native types for wandb
        wandb_scalars = {}
        for key, value in scalars.items():
            # Convert all values to Python scalars
            try:
                # Try to get item() method first (for numpy scalars/0-d arrays)
                if hasattr(value, 'item'):
                    wandb_scalars[key] = value.item()  # type: ignore
                # Try to convert to float for other numeric types
                elif hasattr(value, '__float__'):
                    wandb_scalars[key] = float(value)
                # Fall back to the original value for basic Python types
                else:
                    wandb_scalars[key] = value
            except (AttributeError, ValueError):
                # For arrays with dimensions, provide helpful error
                if hasattr(value, 'shape') and hasattr(value, 'ndim'):
                    if value.ndim > 0:  # type: ignore
                        raise ValueError(f"Expected scalar for key '{key}', got array with shape {value.shape}")  # type: ignore
                # If all else fails, just use the raw value
                wandb_scalars[key] = value
        
        # Log to wandb with the step
        wandb.log(wandb_scalars, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the images occurred.
            images: Mapping from image key to images.
        """
        raise NotImplementedError("write_images is not implemented for WandBWriter yet.")

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """Write videos for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the videos occurred.
            videos: Mapping from video key to videos.
        """
        raise NotImplementedError("write_videos is not implemented for WandBWriter yet.")

    def write_audios(
        self, 
        step: int, 
        audios: Mapping[str, Array], 
        *, 
        sample_rate: int
    ):
        """Write audios for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the audios occurred.
            audios: Mapping from audio key to audios.
            sample_rate: Sample rate for the audios.
        """
        raise NotImplementedError("write_audios is not implemented for WandBWriter yet.")

    def write_texts(self, step: int, texts: Mapping[str, str]):
        """Writes text snippets for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the text snippets occurred.
            texts: Mapping from name to text snippet.
        """
        raise NotImplementedError("write_texts is not implemented for WandBWriter yet.")

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None
    ):
        """Writes histograms for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the arrays were generated.
            arrays: Mapping from name to arrays to summarize.
            num_buckets: Number of buckets used to create the histogram.
        """
        raise NotImplementedError("write_histograms is not implemented for WandBWriter yet.")

    def write_hparams(self, hparams: Mapping[str, Any]):
        """Write hyper parameters.

        Args:
            hparams: Flat mapping from hyper parameter name to value.
        """        
        # Update wandb config with hyperparameters
        wandb.config.update(hparams)

    def flush(self):
        """Tells the MetricWriter to write out any cached values."""
        if self._initialized:
            # wandb doesn't have an explicit flush, but we can ensure sync
            pass

    def close(self):
        """Flushes and closes the MetricWriter."""
        if self._initialized:
            wandb.finish()
            self._initialized = False
