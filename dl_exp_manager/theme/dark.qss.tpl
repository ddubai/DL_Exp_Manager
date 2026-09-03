/* DL Experiment Manager - 테마 스타일시트 템플릿.
   이중 중괄호 자리표시자는 theme/tokens.py 의 값으로 치환된다. 색을 여기에 직접 쓰지 말 것. */

QWidget {
    background-color: {{bg.base}};
    color: {{text.primary}};
}

QMainWindow, QDialog {
    background-color: {{bg.base}};
}

QToolTip {
    background-color: {{bg.elevated}};
    color: {{text.primary}};
    border: 1px solid {{border.strong}};
    border-radius: {{radius.medium}}px;
    padding: 6px 8px;
}

/* ---------- 메뉴 ---------- */
QMenuBar {
    background-color: {{bg.surface}};
    border-bottom: 1px solid {{border.subtle}};
}
QMenuBar::item {
    padding: 5px 10px;
    background: transparent;
    border-radius: {{radius.small}}px;
}
QMenuBar::item:selected { background-color: {{bg.hover}}; }

QMenu {
    background-color: {{bg.elevated}};
    border: 1px solid {{border.subtle}};
    border-radius: {{radius.large}}px;
    padding: 4px;
}
QMenu::item {
    padding: 5px 22px 5px 12px;
    border-radius: {{radius.small}}px;
    color: {{text.primary}};
}
QMenu::item:selected { background-color: {{accent.bg}}; color: {{text.primary}}; }
QMenu::item:disabled { color: {{text.disabled}}; }
QMenu::separator {
    height: 1px;
    background: {{border.subtle}};
    margin: 4px 8px;
}

/* ---------- 입력 ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: {{bg.input}};
    border: 1px solid {{border.default}};
    border-radius: {{radius.medium}}px;
    padding: 4px 8px;
    selection-background-color: {{accent.bg}};
    selection-color: {{text.primary}};
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid {{accent}};
}
QLineEdit:read-only, QPlainTextEdit[readOnly="true"] {
    background-color: {{bg.surface}};
    color: {{text.secondary}};
}
QLineEdit:disabled, QPlainTextEdit:disabled { color: {{text.disabled}}; }

/* ---------- 콤보박스 ---------- */
QComboBox {
    background-color: {{bg.input}};
    border: 1px solid {{border.default}};
    border-radius: {{radius.medium}}px;
    padding: 4px 8px;
    min-height: 20px;
}
QComboBox:hover { border-color: {{border.strong}}; }
QComboBox:focus { border-color: {{accent}}; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border: none;
}
/* 화살표는 Fusion 기본 렌더러에 맡긴다(QSS 로 그리면 플랫폼마다 사각형으로 깨진다).
   색은 QPalette.ButtonText 를 따라가므로 다크에서도 잘 보인다. */
QComboBox QAbstractItemView {
    background-color: {{bg.elevated}};
    border: 1px solid {{border.strong}};
    border-radius: {{radius.large}}px;
    padding: 4px;
    outline: none;
    selection-background-color: {{accent.bg}};
    selection-color: {{text.primary}};
}
QComboBox QAbstractItemView::item {
    padding: 5px 8px;
    border-radius: {{radius.small}}px;
    min-height: 20px;
}
QComboBox QAbstractItemView::item:hover { background-color: {{bg.hover}}; }

/* ---------- 버튼 ---------- */
QPushButton {
    background-color: {{bg.elevated}};
    border: 1px solid {{border.default}};
    border-radius: {{radius.medium}}px;
    padding: 6px 14px;
    color: {{text.primary}};
}
QPushButton:hover { background-color: {{bg.hover}}; border-color: {{border.strong}}; }
QPushButton:pressed { background-color: {{bg.surface}}; }
QPushButton:disabled { color: {{text.disabled}}; border-color: {{border.subtle}}; }
QPushButton[variant="primary"] {
    background-color: {{accent}};
    border: 1px solid {{accent}};
    color: {{text.on-accent}};
    font-weight: 600;
}
QPushButton[variant="primary"]:hover { background-color: {{accent.hover}}; border-color: {{accent.hover}}; }
QPushButton[variant="primary"]:pressed { background-color: {{accent.pressed}}; }
QPushButton[variant="danger"] {
    background-color: transparent;
    border: 1px solid {{danger}};
    color: {{danger}};
}
QPushButton[variant="danger"]:hover { background-color: {{bg.hover}}; }

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {{radius.medium}}px;
    padding: 4px 8px;
    color: {{text.secondary}};
}
QToolButton:hover {
    background-color: {{bg.hover}};
    border-color: {{border.default}};
    color: {{text.primary}};
}
QToolButton:pressed { background-color: {{bg.surface}}; }
QToolButton::menu-indicator { image: none; }

/* ---------- 표 ---------- */
QTableView, QTreeWidget, QTreeView, QListView {
    background-color: {{bg.surface}};
    alternate-background-color: {{bg.surface.alt}};
    border: 1px solid {{border.subtle}};
    border-radius: {{radius.medium}}px;
    gridline-color: {{border.subtle}};
    outline: none;
    selection-background-color: {{accent.bg}};
    selection-color: {{text.primary}};
}
QTableView::item { padding: 2px 6px; border: none; }
QTableView::item:hover { background-color: {{bg.hover}}; }
QTableView::item:selected { background-color: {{accent.bg}}; color: {{text.primary}}; }

QTreeWidget::item, QTreeView::item, QListView::item {
    padding: 3px 4px;
    border-radius: {{radius.small}}px;
}
QTreeWidget::item:hover, QTreeView::item:hover, QListView::item:hover {
    background-color: {{bg.hover}};
}
QTreeWidget::item:selected, QTreeView::item:selected, QListView::item:selected {
    background-color: {{accent.bg}};
    color: {{text.primary}};
}

QHeaderView::section {
    background-color: {{bg.elevated}};
    color: {{text.secondary}};
    padding: 5px 8px;
    border: none;
    border-right: 1px solid {{border.subtle}};
    border-bottom: 1px solid {{border.strong}};
    font-weight: 600;
}
QHeaderView::section:hover { background-color: {{bg.hover}}; color: {{text.primary}}; }
QHeaderView::down-arrow, QHeaderView::up-arrow { width: 9px; height: 9px; }
QTableCornerButton::section {
    background-color: {{bg.elevated}};
    border: none;
    border-bottom: 1px solid {{border.strong}};
}

/* ---------- 탭 ---------- */
QTabWidget::pane {
    border: 1px solid {{border.subtle}};
    border-radius: {{radius.medium}}px;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    color: {{text.secondary}};
    padding: 7px 16px;
    margin-right: 2px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:hover { color: {{text.primary}}; }
QTabBar::tab:selected {
    color: {{accent}};
    border-bottom: 2px solid {{accent}};
    font-weight: 600;
}

/* ---------- 그룹박스 ---------- */
QGroupBox {
    background-color: {{bg.surface}};
    border: 1px solid {{border.subtle}};
    border-radius: {{radius.large}}px;
    margin-top: 14px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {{text.secondary}};
    font-weight: 600;
}

/* ---------- 스크롤바 ---------- */
QScrollBar:vertical {
    background: transparent;
    width: 11px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: {{border.strong}};
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: {{text.disabled}}; }
QScrollBar:horizontal {
    background: transparent;
    height: 11px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: {{border.strong}};
    border-radius: 5px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: {{text.disabled}}; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---------- 스플리터 ---------- */
QSplitter::handle { background-color: {{border.subtle}}; }
QSplitter::handle:hover { background-color: {{accent.border}}; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }

/* ---------- 기타 ---------- */
QScrollArea { border: none; background-color: {{bg.base}}; }
QStatusBar {
    background-color: {{bg.surface}};
    border-top: 1px solid {{border.subtle}};
    color: {{text.secondary}};
}
QStatusBar::item { border: none; }
QLabel { background: transparent; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid {{border.strong}};
    border-radius: {{radius.small}}px;
    background-color: {{bg.input}};
}
QCheckBox::indicator:hover { border-color: {{accent}}; }
QCheckBox::indicator:checked {
    background-color: {{accent}};
    border-color: {{accent}};
}
QCheckBox::indicator:disabled { border-color: {{border.subtle}}; background-color: {{bg.surface}}; }
QSizeGrip { background: transparent; }
