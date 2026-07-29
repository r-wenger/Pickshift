import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

PLUGIN_DIR = os.path.dirname(__file__)


class PickShiftPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon = QIcon(os.path.join(PLUGIN_DIR, "icon.svg"))
        self.action = QAction(icon, "PickShift...", self.iface.mainWindow())
        self.action.setToolTip(
            "Monte-Carlo estimation of positional and area uncertainty from GCP biases"
        )
        self.action.triggered.connect(self.run)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToVectorMenu("&PickShift", self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removePluginVectorMenu("&PickShift", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if self.dialog is not None:
            self.dialog.close()
            self.dialog = None

    def run(self):
        from .pickshift_dialog import PickShiftDialog

        if self.dialog is None:
            self.dialog = PickShiftDialog(self.iface, self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
