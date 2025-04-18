import contextlib
import platform
import uuid
import warnings
import weakref
import random
import string
from collections import defaultdict
from typing import (
    Any,
    Callable,
    ContextManager,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)
from typing import *  # noqa: F403
from weakref import ReferenceType

import torch
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode

__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "matepoint_pipeline",
    "CheckpointError",
    "check_backward_validity",
    "detach_variable",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
    "set_checkpoint_debug_enabled",
    "CheckpointPolicy",
    "SelectiveCheckpointContext",
    "create_selective_checkpoint_contexts",
    "SAC_IGNORED_OPS",
]

_DEFAULT_DETERMINISM_MODE = "default"

_checkpoint_debug_enabled: Optional[bool] = None

class StreamManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stream = torch.cuda.Stream()
        return cls._instance

class NestedMatepointContext:
    _instance = None
    _INSIDE_MATEPOINT = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NestedMatepointContext, cls).__new__(cls)
        return cls._instance

    def __enter__(self):
        assert not NestedMatepointContext._INSIDE_MATEPOINT, "Nested matepoint calls are not supported"
        NestedMatepointContext._INSIDE_MATEPOINT = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        NestedMatepointContext._INSIDE_MATEPOINT = False

@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled: Optional[bool]):
    """
    Context manager that sets whether checkpoint should print additional debug
    information when running. See the ``debug`` flag for
    :func:`~torch.utils.checkpoint.checkpoint` for more information. Note that
    when set, this context manager overrides the value of ``debug`` passed to
    checkpoint. To defer to the local setting, pass ``None`` to this context.

    Args:
        enabled (bool): Whether checkpoint should print debug information.
            Default is 'None'.
    """
    global _checkpoint_debug_enabled
    try:
        prev = _checkpoint_debug_enabled
        _checkpoint_debug_enabled = enabled
        yield
    finally:
        _checkpoint_debug_enabled = prev


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None"
        )


class DefaultDeviceType:
    r"""
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """

    _default_device_type = "cuda"

    @staticmethod
    def set_device_type(device: str = "cuda"):
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        return DefaultDeviceType._default_device_type


def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg):
        nonlocal device_types
        if isinstance(arg, torch.Tensor) and arg.device.type not in {"cpu", "meta"}:
            device_types.append(arg.device.type)
    tree_map(add_device_types, args)

    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}"
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "cuda" in device_types_set:
        return "cuda"
    else:
        return device_types[0]


def _get_autocast_kwargs(device_type="cuda"):
    if torch.amp.is_autocast_available(device_type):
        device_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None

    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled('cpu'),
        "dtype": torch.get_autocast_dtype('cpu'),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return device_autocast_kwargs, cpu_autocast_kwargs


#Gmatepoint_stream = None
Gmatepoint_ctx = []

def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()

# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@torch._disable_dynamo
def checkpoint(
    function,
    *args,
    stream = None,
    matepoint_ctx = None,
    use_reentrant: Optional[bool] = False,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    pipeline: bool = True,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        checkpointed version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.5 we will raise an exception if ``use_reentrant`` is not passed. If you
        are using the ``use_reentrant=True`` variant, please refer to the note below
        for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function.
            Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    with NestedMatepointContext():
        stream = StreamManager().stream # Matepoint sucks without it's own stream
        if matepoint_ctx is None:
            matepoint_ctx = Gmatepoint_ctx
        if use_reentrant is None or use_reentrant is True:
            warnings.warn(
                "Overriding to set use_reentrant=False as no recomputation happens in matepoint",
                stacklevel=2
            )

        # Hack to mix *args with **kwargs in a python 2.7-compliant way
        preserve = kwargs.pop("preserve_rng_state", True)
        if preserve:
            warnings.warn(
                "Overriding to set preserve_rng_state=False as no recomputation happens in matepoint",
                stacklevel=2
            )
        check_inplace_modifications_flag = kwargs.pop("check_inplace_modifications", False)
        if kwargs:
            raise ValueError(
                "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
            )

        gen = _checkpoint_without_reentrant_generator(
            function, stream, matepoint_ctx, context_fn, determinism_check, debug, check_inplace_modifications_flag, pipeline, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret


def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will not store
    the intermediate  activations. The inputs of each checkpointed segment will
    be saved for re-running the segment in the backward pass.

    .. warning::
        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.5 we will raise an exception if ``use_reentrant`` is not passed. If you
        are using the ``use_reentrant=True` variant, please see
        :func:`~torch.utils.checkpoint.checkpoint` for
        the important considerations and limitations of this variant. It is
        recommended that you use ``use_reentrant=False``.

    .. warning::
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function.
            Default: ``True``

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(
            run_function(start, end, functions),
            input,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve,
        )
    return run_function(end + 1, len(functions) - 1, functions)(input)


def _internal_assert(cond):
    if not cond:
        raise AssertionError(
            "Something went unexpectedly wrong in activation checkpoint. "
            "Please report this bug by filing an issue to PyTorch."
        )


# NOTE [ Nestable Checkpoint ]
#
# The semantics of nested checkpoint can be defined by two basic rules.
# Following the two rules leads to an important implication that is central
# to motivating the design.
#
# Rule 1. Saved tensors are managed by inner-most checkpoint only and hidden
#         from any outer layers of checkpoint.
#
# Rule 2. The inputs of inner checkpoints are treated as tensors saved to its
#         parent checkpoint.
#
# Implication: To recompute any given saved tensor, we need to recompute all of
#              the checkpoints wrapping it.
#
# Why is this implied? To unpack a saved tensor X during backward we need to
# recompute the inner-most checkpoint (#1), and in order to recompute that
# checkpoint I need to have its inputs, which are managed by that checkpoint's
# parent (#2), which thus also needs to be recomputed first. Continue this line
# of reasoning and we realize that in order to unpack X, all checkpoints that
# were active at the time X was saved need to be recomputed. (unless we have
# already done so in that backward for some other saved tensor).
#
# In practice, we use a noop autograd Function to save inputs as saved tensors.
# During unpack calling ctx.saved_tensor triggers the parent checkpoint to
# recompute.
#
# Rule 3. We should start recomputation as if there are no checkpoints currently
#         active. Checkpoints encountered during recomputation are still
#         respected.
#
# When we start recomputation, we push the saved variable hook meant for
# recomputation on the stack. See examples in Rule 6 for more context.
#
#                                  * * * *
#
# Beyond the basic semantics specific to nested checkpoint, we impose several
# more constraints that may apply to checkpointing in general.
#
# Rule 4. Lifetime of recomputed tensors
#
#         Recomputed tensors are considered specific to particular invocations
#         of backward and are always cleared immediately as they are unpacked
#         Particularly, we require this to happen even if retain_graph=True.
#
# [ Implementation details of Rule 4 ]
#
# If we were okay with recomputed tensors staying alive after backward is run
# with retain_graph=True, we would store recomputed variables as the values of a
# WeakKeyDictionary and pack strong references to the keys, so that as we
# backward, those packed keys would be cleared as long as retain_graph=False.
# Clearing the packed key clears the corresponding entry in the WKD.
#
# If we wish recomputed variables to be immediately cleared as we unpack them in
# the retain_graph=True case, we cannot rely on the packed keys to be cleared by
# backward automatically. Instead of packing the strong reference to the key
# directly, we pack a container object, which we manually clear as we unpack.
#
# An important detail is that if a second backward happens, the second
# recomputation needs to reset the container with a newly created key.
#
# Rule 5. Stop recomputation as soon as we've recomputed the saved tensors we
#         know we need.
#
# [ Implementation details of Rule 5 ]
#
# During recomputation, raise an exception if the number of recomputed tensors
# matches the number of tensors that we expected to recompute. We wrap the
# recomputation call with a try-catch to catch this specific exception. See
# Rule #6 below for some examples.
#
# Rule 6. We support doing backward inside checkpoint context
#
# [ retain_graph is True]
#
# def fn(x):
#   y = x.sin()
#   z = y.cos()
#   gx, = torch.autograd.grad(z, x, retains_grad=True)
#   return gx, z
#
# out = checkpoint(fn)(inp)
# out.backward()
#
# Because z is saved by cos while checkpoint is enabled, it would not be
# actually saved, and so the .grad() call inside must trigger a recomputation.
#
# During recomputation the "inner pack hook" has two responsibilities:
#
# 1) As usual, populating the WeakKeyDictionary storing recomputed tensors
# 2) Pack the actual tensor (detached) so that one may perform backward on the
#    recomputed graph. The tensors saved to this graph will live until the end
#    of recomputation, or die earlier if someone performs backward with
#    retain_graph=False.
#
# More generally performing backward on the recomputed graph occurs in the
# following cases:
# - If backward is performed inside forward,
#   - During the original forward IF early-stop is disabled
#   - During the original backward
# - If there are multiple .grad()/.backward() calls, we would perform backward
#   on the recomputed graph even if early-stop is enabled (see the example below)
#
# [ retain_graph is False ]
#
# The example below shows what happens if during recomputation we find that some
# of the tensors we are trying to recompute have already been cleared.
#
# Spoiler: we don't do anything special, we just skip over them!
#
# def fn(x):
#   y = x.sin()                           # (1)
#   z = y.cos()                           # (2)
#   gx, = torch.autograd.grad(z, x)       # (3)
#   return x.cos() * gx                   # (4)
#
# out = checkpoint(fn)(inp)
# out.backward()                          # (5)
#
# 1, 2. Don't save x and y since we are inside a checkpoint.
# 3. Trigger a recompute of fn since x and y weren't saved.
#    And depending on whether early stop is enabled, either stop at (2) or
#    continue running the function.
#    Because we are running backward with retain_graph=False, we clear x and y's
#    holders.
# 4. Don't save x since we are inside a checkpoint.
# 5. Calling backward triggers another recompute of fn. During recompute, we see
#    that x and y have already been cleared in the original graph as indicated
#    by holder=None. We skip over them. We still save x at (4) (since its holder,
#    is still alive.)

_enable_checkpoint_early_stop = True


@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    """Context manager that sets whether checkpoint should stop recomputation
    early.

    By default, non-reentrant checkpoint stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.

    Example::

    >>> # xdoctest: +SKIP(failing)
    >>> message = "saved tensors default hooks are disabled"
    >>> with set_checkpoint_early_stop(False):
    ...     # Any checkpoint under this context manager will respect this
    ...     # context manager, even if its backward is performed outside.
    ...     out = checkpoint(fn, inputs)
    ...
    >>> out.backward()
    """
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev


class _Handle:
    pass


class _Holder:
    def __init__(self):
        self.handles: Dict[int, Optional[_Handle]] = dict()


class _NoopSaveInputs(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        return torch.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        # Only tensors can be saved with ctx.save_for_backward, everything else
        # is captured by get_args, which is saved directly on ctx
        tensor_indices, tensors = zip(
            *[(i, o) for i, o in enumerate(inputs) if isinstance(o, torch.Tensor)]
        )
        idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}
        # args but with tensors replaced with None as placeholders
        args = [None if isinstance(o, torch.Tensor) else o for o in inputs]

        def get_args(saved_tensors):
            # restore the placeholders with the original tensors grabbed from
            # ctx.saved_tensors (which may be saved on a parent checkpoint if
            # this checkpoint is nested, and that would trigger a recursive
            # unpack!)
            ret = [
                saved_tensors[idx2saved_idx[i]] if i in tensor_indices else o
                for i, o in enumerate(args)
            ]
            # grab the tail since we also saved the dummy to avoid having to explicitly
            # handle the case where there are no tensor inputs
            return ret[1:]

        ctx.get_args = get_args
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise AssertionError("Did not expect to backward on this graph")


class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn):
        self.recompute_fn = recompute_fn
        self.input_saver = None
        self.weak_holders: List[ReferenceType] = []
        # We store this as a weakkeydictionary so that in the case of a partial
        # backward, the entries in the dict are cleared alongside the Holder
        # which will be removed when the SavedVariable is cleared.
        self.recomputed: DefaultDict[
            int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)
        # We need both recomp_counter and recomputed since they can diverge
        # https://github.com/pytorch/pytorch/pull/90105#discussion_r1135889885
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)

        # See Rule 5
        self.early_stop = early_stop

        # Debugging
        self.metadata_fn = metadata_fn
        self.unpack_error_cb = unpack_error_cb
        self.x_metadatas = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False

    def check_recomputed_tensors_match(self, gid):
        if self.ignore_saved_mismatch:
            # TODO: we can probably make this check stricter by checking that
            #       the metadata of the first tensors still match.
            return
        # NOTE [ Error handling for checkpoint ]
        #
        # At a high level, we need to check that the tensors saved
        # during original forward matches tensors saved during recompute
        # This means handling 3 cases:
        #
        # 1. During recompute, more tensors were saved.
        #
        #    Usually this is hidden due to the StopRecomputationError
        #    but if early stop is not enabled, or we would have errored
        #    anyway because there aren't enough weak_holders. But we
        #    do want to have a nice error. See the _recomputation_hook
        #    for details.
        if not len(self.weak_holders) == self.recomp_counter[gid]:
            # 2. During recompute, fewer tensors were saved
            #
            # We know that everytime we save something do original forward
            # we append to weak_holder, and every time we save a tensor
            # during recompute we increment recompute_counter.
            raise CheckpointError(
                "torch.utils.checkpoint: A different number of tensors was saved "
                "during the original forward and recomputation.\n"
                f"Number of tensors saved during forward: {len(self.weak_holders)}\n"
                f"Number of tensors saved during recomputation: {self.recomp_counter[gid]}"
            )

        # 3. During recompute, the same tensors were saved, but they
        #    have different metadata
        nb_meta_different = []
        for idx, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            # We've seen all holders since we iterate over them in order
            # For every holder that is still alive now, it must've been
            # alive when we saw it during recompute, therefore, the
            # gid must be set.
            _internal_assert(gid in holder.handles)
            # We know this is the first unpack, so it couldn't have been set
            # to None yet.
            _internal_assert(holder.handles[gid] is not None)
            # We always set these together in the recomputation hook
            _internal_assert(holder.handles[gid] in self.recomputed[gid])
            # see pack hook, x_metadata is 1:1 with weak_holders.
            x_meta = self.x_metadatas[idx]
            recomputed_x = self.recomputed[gid][holder.handles[gid]]
            if x_meta != self.metadata_fn(recomputed_x):
                nb_meta_different.append((idx, x_meta, self.metadata_fn(recomputed_x)))

        if len(nb_meta_different) > 0:
            mismatched_tensors = ""
            for idx, x_meta, recomputed_meta in nb_meta_different:
                mismatched_tensors += (
                    f"tensor at position {idx}:\n"
                    f"saved metadata: {x_meta}\n"
                    f"recomputed metadata: {recomputed_meta}\n"
                )
            raise CheckpointError(
                "torch.utils.checkpoint: Recomputed values for the following tensors "
                "have different metadata than during the forward pass.\n"
                f"{mismatched_tensors}"
            )


_checkpoint_error_template = """ \
An error happened while unpacking tensors; dumping logs of latest computation
because you passed `debug=True` to `torch.utils.checkpoint.checkpoint()`.
Scroll all the way down for guidance on how to navigate these logs.

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        1. Stack traces of the operators that ran in the original forward     |
+------------------------------------------------------------------------------+

{forward_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        2. Stack traces of the operators that ran during recomputation        |
+------------------------------------------------------------------------------+

{recompute_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|       3. Log of operators in the original forward and recomputation          |
+------------------------------------------------------------------------------+
(Scroll up to correlate stack traces with each operation listed below. This
 helps identify their source in the code.)

IMPORTANT: Differences in "detach" calls between the original forward and the
           recomputation are expected. They are introduced by the checkpointing
           mechanism and can be ignored.

Operations executed during the original forward:

{forward_ops}

Operations executed during recomputation:

{recompute_ops}

+------------------------------------------------------------------------------+
 ERROR: Detected non-determinism while running activation checkpointing

 You are seeing this error because you passed `debug=True` to checkpoint and
 tensors to be saved during the original forward and differ between those saved
 during recomputation. This can happen if different operators were ran in the
 original forward and in the recomputation.

 To identify where the mismatch may be coming from, you can do the following:

 1) Compare the operators ran during original forward and recomputation to
    see where they differ. These operators are printed above in the order they
    were executed.

 2) Review the stack trace for each operator to locate its invocation source.
    Each operator's stack trace is printed in their execution order.

 Note that the logs can be quite long. Here's how they are structured:
 (Tip: you can Ctrl-f for these headers)

 1. Stack traces of the operators that ran in the original forward
 2. Stack traces of the operators that ran during recomputation
 3. Log of operators in the original forward and recomputation
 4. Error message                                             <--- You are here
--------------------------------------------------------------------------------
"""

class CheckpointError(RuntimeError):
    pass


def _get_debug_context_and_cb() -> Tuple[Callable[[], Any], Callable[[CheckpointError], None]]:
    # This function returns the context_fn and error_cb to be used by the
    # checkpointing mechanism. error_cb is invoked when an error is detected
    # during unpack.

    # record_context_cpp is not support on non-linux non-x86_64 platforms
    cpp_tb = platform.machine() == 'x86_64' and platform.system() == 'Linux'

    class CaptureLogs:
        def __init__(self):
            self.logs = None
            self.tbs = None

        def get_context_manager(self):
            @contextlib.contextmanager
            def logging_mode():
                with LoggingTensorMode(), \
                     capture_logs(True, python_tb=True, script_tb=True, cpp_tb=cpp_tb) as logs_and_tb:
                    self.logs, self.tbs = logs_and_tb
                    yield logs_and_tb
            return logging_mode()

    capture_logs_fwd = CaptureLogs()
    capture_logs_recompute = CaptureLogs()

    def unpack_error_cb(e: CheckpointError):
        def get_str_tb(label, capture_logs):
            out = ""
            total_len = len(capture_logs.logs)
            for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs)):
                out += f"{log}   ({i + 1} of {total_len} in {label})\n\n"
                found_torch_dispatch = False
                for line in tb:
                    # Start printing stack trace only after __torch_dispatch__ is found
                    is_torch_dispatch = line['name'] == '__torch_dispatch__'
                    if not found_torch_dispatch and not is_torch_dispatch:
                        continue
                    elif is_torch_dispatch:
                        found_torch_dispatch = True
                        continue
                    out += f"{line['filename']}:{line['line']}:{line['name']}\n"
                out += "\n\n"
            return out
        assert capture_logs_fwd.logs is not None
        assert capture_logs_recompute.logs is not None
        raise CheckpointError(
            _checkpoint_error_template.format(
                forward_traces=get_str_tb("original", capture_logs_fwd),
                recompute_traces=get_str_tb("recompute", capture_logs_recompute),
                forward_ops="\n".join(capture_logs_fwd.logs),
                recompute_ops="\n".join(capture_logs_recompute.logs)
            )
        ) from e

    def context_fn():
        return capture_logs_fwd.get_context_manager(), capture_logs_recompute.get_context_manager()

    return context_fn, unpack_error_cb


def _default_meta_extractor(x: torch.Tensor) -> Dict[str, Any]:
    # These properties are fast to check, easy to understand
    return {
        "shape": x.shape,
        "dtype": x.dtype,
        "device": x.device
    }

_allowed_determinism_checks_to_fns: Dict[str, Callable[[torch.Tensor], Any]] = {
    _DEFAULT_DETERMINISM_MODE: _default_meta_extractor,
    "none": lambda _: None,
}

# See Rule 5
class _StopRecomputationError(Exception):
    pass


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: int):
        def pack_hook(x):
            x = x.detach() if x.requires_grad else x
            target_frame = target_frame_ref()
            assert target_frame is not None  # appease mypy
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            if recomp_idx >= len(target_frame.weak_holders):
                assert not target_frame.early_stop
                if not target_frame.forward_completed:
                    # We run into this case when early stop is not enabled and do
                    # grad within checkpoint.
                    # We need to set this flag, so we don't error out later when
                    # we check if the number of tensors saved during forward and
                    # recomputation match.
                    target_frame.ignore_saved_mismatch = True
                    return x
                raise CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass."
                )

            holder = target_frame.weak_holders[recomp_idx]()

            # This holder may have been cleared because someone may have called
            # backward within forward. If so, we don't need to save.
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x

            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError()
            # See Rule 6: [ retain_graph is True ] above
            return x

        def unpack_hook(x):
            # See Rule 6: [ retain_graph is True ] above for an example of when
            # the graph created during recomputation could be backwarded.
            return x

        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                # generate a temporary id if we trigger unpack outside of a backward call
                gid = int(uuid.uuid4())

            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)

                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)

            _internal_assert(gid in holder.handles)

            if holder.handles[gid] is None:
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                    "so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)


# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.

# Check for environment variable to control printing
PRINT = False
import os
if os.environ.get('MATEPOINT_PRINTS', '') == '1':
    PRINT = True

def size_mb(t):
    return t.numel() * t.element_size() / 1024 / 1024

# This function is called when we are sending tensors back to the GPU in the backwards pass
def matepoint_pipeline(ctx, stream):
    if len(ctx) >= 1:
        with torch.cuda.stream(stream):
            if 1:
                ctx_extra = []
                check = ctx[-1][-1]
                for x, d, _ in zip(*ctx[-1]):
                    if hasattr(x, "device"):
                        if PRINT:
                            print(f"[=>] moving (p) {[xx for xx in x.shape]} {size_mb(x):.2f} MiB from {x.device} to {d} | check: {check}")
                        ctx_extra.append(x.to(d, non_blocking=True))
                    else:
                        ctx_extra.append(x)
                ctx[-1][0] = ctx_extra
            else:
                ctx[-1][0] = [x.to(d, non_blocking=True) if hasattr(x, "device") else x for x, d, _ in zip(*ctx[-1])]

import hashlib
def elist(xx):
    # return
    out = []
    for x in xx:
        # out.append(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()[:4])
        out.append(x.flatten()[0].item())
    return out

bigTable = None
bigN = 0

def check_inplace_modifications(tensor):
    """Check if a tensor has been modified in-place by comparing its version counter."""
    if hasattr(tensor, "_version"):
        current_version = tensor._version
        if current_version > 0:
            raise RuntimeError(
                f"One of the variables has been modified by an inplace operation: "
                f"[{tensor.type()} {list(tensor.shape)}], which is at version {current_version}; "
                f"expected version 0 instead. In-place modifications are not supported with matepoint."
            )

def _checkpoint_without_reentrant_generator(
    fn,
    stream,
    matepoint_ctx=None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    check_inplace_modifications_flag: bool = False,
    pipeline: bool = True,
    *args,
    **kwargs
):
    """Checkpointing without reentrant autograd

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    global bigN, bigTable

    # global all_contexts

    ogextra = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))

    unpack_error_cb = None

    if debug:
        if context_fn != noop_context_fn:
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device = _infer_device_type(*args)
    forward_context, recompute_context = context_fn()
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device)

    def recompute_fn(*inputs):
        kwargs, *args = inputs
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
    
        device_autocast_ctx = torch.amp.autocast(
            device_type=device, **device_autocast_kwargs
        ) if torch.amp.is_autocast_available(device) else contextlib.nullcontext()
        with device_autocast_ctx, torch.amp.autocast(device_type="cpu",**cpu_autocast_kwargs), \
                recompute_context:
            if not pipeline:
                idx = [i for i in range(len(matepoint_ctx)) if matepoint_ctx[i][2] == ogextra]
                assert len(idx) == 1
                idx = idx[0]
                # print("fetched", idx)
                args, devices, checksum = matepoint_ctx.pop(idx)
            else:
                args, devices, checksum = matepoint_ctx.pop()

            # This is where tensors are moved back to the GPU for computation in the backwards pass.
            #

            if 1:
                newargs = []
                for arg, d in zip(args, devices):
                    if callable(arg):  # bigTable stuff, fix indexing
                        arg = arg()[None]
                    if d is not None:
                        if arg.device != d:
                            if PRINT:
                                print(f"[=>] moving (p) {[aa for aa in arg.shape]} {size_mb(arg):.2f} MiB from {arg.device} to {d} | check: {checksum}, og: {ogextra}")
                            arg = arg.to(d)
                    newargs.append(arg)
                args = newargs

            else:  # keeping the old list comprehension for reference
                args = [(arg.to(d) if arg.device != d else arg) if d is not None else arg for arg, d in zip(args, devices)]

            assert checksum == ogextra, f"matepoint checksum mismatch: {checksum} vs {ogextra}"
            if pipeline:
                with torch.cuda.stream(stream):
                    torch.cuda.synchronize()
                    if matepoint_ctx is not None:

                        # Most tensors should be moved in the pipelined move
                        matepoint_pipeline(matepoint_ctx, stream)

            fn(*[x for x, d in zip(args, devices)], **kwargs)

    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop,
        unpack_error_cb,
        metadata_fn
    )
    dummy = torch.empty((0,), requires_grad=True)
    cpu = torch.device("cpu")
    # kwargs["devices"] = [x.device if hasattr(x, "device") else None for x in args]
    args_cpu, devices = [], []
    with torch.cuda.stream(stream):

        for arg in args:
            device_ = arg.device if hasattr(arg, "device") else None
            devices.append(device_)
            torch.cuda.synchronize()

            # This part is called during the forwards pass, and is when tensors are sent back to the CPU

            if device_ is not None:
                if PRINT:
                    print(f"[<=] moving {[aa for aa in arg.shape]} {size_mb(arg):.2f} MiB from {arg.device} to {cpu} | og: {ogextra}")
                    # print(MemoryMonitor().str())
                if bigTable is not None and arg[0].shape[-1] == bigTable[0].shape[-1]:
                    bigTable[bigN].copy_(arg[0], non_blocking=True)
                    def oops(capt):
                        def fn():
                            return bigTable[capt]
                        return fn
                    xcpu = oops(bigN)
                    bigN += 1
                else:
                    xcpu = arg.to(cpu, non_blocking=True)
            else:
                xcpu = arg
            args_cpu.append(xcpu)

    new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, len(matepoint_ctx))
    matepoint_ctx.append([args_cpu, devices, ogextra])
    del xcpu
    del args_cpu

    # When ambient grad_mode is False
    if new_frame.input_saver.grad_fn is None:
        yield
        return

    with _checkpoint_hook(new_frame), forward_context:
        yield
    if check_inplace_modifications_flag:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                check_inplace_modifications(arg)
    new_frame.forward_completed = True

    return
