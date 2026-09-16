"""Colours and the application stylesheet."""

BG = "#111827"
PANEL = "#1F2937"
PANEL2 = "#273244"
BORDER = "#374151"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#60A5FA"
ACCENT2 = "#93C5FD"
OK = "#34D399"
WARN = "#FBBF24"
ERR = "#F87171"
STALE = "#F59E0B"
BLOCKED = "#6B7280"
OVERRIDE = "#93C5FD"

STATE_COLORS = {
    "not started": BLOCKED,
    "partly done": ACCENT,
    "done": OK,
    "stale": STALE,
    "errors": ERR,
    "blocked": BLOCKED,
}

QSS = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QMainWindow::separator {{ background: {BORDER}; width: 2px; height: 2px; }}
QFrame#Card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#SubCard {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLabel#Title {{ font-size: 20px; font-weight: 600; color: {TEXT}; }}
QLabel#Subtitle {{ font-size: 13px; color: {MUTED}; }}
QLabel#Hint {{ color: {MUTED}; font-size: 12px; }}
QLabel#Mono {{ font-family: Consolas, 'DejaVu Sans Mono', monospace; }}
QLabel#Section {{ font-size: 14px; font-weight: 600; color: {ACCENT2}; margin-top: 6px; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px; selection-background-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {MUTED}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {PANEL2}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {PANEL}; }}
QPushButton#Primary {{ background: #1D4ED8; border-color: #2563EB; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #2563EB; }}
QPushButton#Danger {{ background: #7F1D1D; border-color: #991B1B; }}
QPushButton#Danger:hover {{ background: #991B1B; }}
QPushButton#Flat {{ background: transparent; border: none; color: {ACCENT2}; padding: 2px 6px; }}
QToolButton#Collapser {{ background: transparent; border: none; color: {ACCENT2}; padding: 2px 4px; font-weight: 600; }}
QToolButton#Collapser:hover {{ text-decoration: underline; }}
QPushButton#Flat:hover {{ text-decoration: underline; }}
QListWidget#Nav {{ background: {PANEL}; border: none; font-size: 14px; outline: 0; }}
QListWidget#Nav::item {{ padding: 10px 12px; border-left: 3px solid transparent; }}
QListWidget#Nav::item:selected {{ background: {PANEL2}; border-left: 3px solid {ACCENT}; color: {TEXT}; }}
QListWidget#Nav::item:hover {{ background: {PANEL2}; }}
QTreeWidget, QTreeView, QListWidget, QTableWidget {{ background: {PANEL}; border: 1px solid {BORDER}; alternate-background-color: {PANEL2}; }}
QHeaderView::section {{ background: {PANEL2}; border: none; border-bottom: 1px solid {BORDER}; padding: 4px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: #1E3A5F; }}
QProgressBar {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 12px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT2}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 4px; }}
QTabBar::tab {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 6px 12px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {PANEL2}; border-bottom-color: {PANEL2}; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QDockWidget::title {{ background: {PANEL}; padding: 4px; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: {PANEL2}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {PANEL2}; }}
QStatusBar {{ background: {PANEL}; }}
"""
