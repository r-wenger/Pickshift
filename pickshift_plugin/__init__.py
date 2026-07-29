def classFactory(iface):
    from .pickshift_plugin import PickShiftPlugin
    return PickShiftPlugin(iface)
