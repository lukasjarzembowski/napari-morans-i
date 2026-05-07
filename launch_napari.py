"""Convenience launcher for debugging the Moran's I plugin in an IDE.

This script follows the pattern recommended in the napari "Debugging during
plugin development" guide:
    https://napari.org/dev/plugins/building_a_plugin/debug_plugins.html#quick-start

Run it instead of bare ``napari`` when you want to:

* set IDE breakpoints inside the plugin code,
* drop straight into the dock widget without clicking through menus,
* preload a sample image so the plugin is ready to run on startup.

Usage
-----
    python launch_napari.py
"""

from __future__ import annotations

from napari import Viewer, run

from napari_morans_i._sample_data import make_sample_data


def main() -> None:
    viewer = Viewer()

    # Preload the bundled sample image so the plugin has something to run on.
    image, kwargs, _ = make_sample_data()[0]
    viewer.add_image(image, **kwargs)

    # Open the dock widget straight away. The first argument is the *plugin*
    # name (the value of ``name:`` in ``napari.yaml``); the second is the
    # widget's ``display_name``.
    viewer.window.add_plugin_dock_widget(
        'napari-morans-i', "Moran's I Analysis"
    )

    run()


if __name__ == '__main__':
    main()
