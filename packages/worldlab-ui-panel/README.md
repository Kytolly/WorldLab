# worldlab-ui-panel

Optional read-only Panel UI for WorldLab runtime events.

Install the core package and this UI package with:

```text
pip install "worldlab[ui]"
```

Run the synthetic ExampleWorldModel closed loop:

```text
worldlab-panel --goal 8 --step-delay 0.6
```

The page reads a `worldlab.TraceSource` and displays the current runtime
status, total reward, chunk index, model latency, latest multi-view synthetic
frame, action/state matrices, and the read-only event timeline. The Panel
package does not call or control the environment, policy, or world model.

For an in-process application, create the view from an existing event source:

```python
from worldlab import EventBuffer
from worldlab_ui_panel import create_panel

source = EventBuffer()
view = create_panel(source)
```

Pass `view` to `panel.serve`. A future transport reader can implement the same
`TraceSource` protocol without changing the UI.
