"""omniact: the OmniAct cross-embodiment policy and its closed-loop evaluation client.

The policy (:mod:`omniact.model`, trained by :mod:`omniact.trainer` on :mod:`omniact.dataset`) represents
the robot as a per-node geometric scene graph: each joint/fingertip is a node with (center, axis,
connections), read from the WODS ``node_geometry`` / ``node_graph`` fields, and the output is per-node
relative joint coordinates. It is **embodiment-agnostic**: the node set comes entirely from the data, with
no robot-specific code or hardcoded node count. The sim/FK machinery that produces that data lives in
omniact-molmospaces, which also serves the simulator that :mod:`omniact.rollout` drives over gRPC for
closed-loop evaluation. Import submodules directly."""
