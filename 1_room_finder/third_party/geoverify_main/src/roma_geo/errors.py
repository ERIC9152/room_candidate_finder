"""可判讀的失敗：指出是哪一項輸入、哪一個條件不成立。"""


class RomaGeoError(Exception):
    """流程失敗。`input_name` 是缺席或不合格的輸入，`condition` 是未滿足的條件。"""

    def __init__(self, message, input_name=None, condition=None):
        self.input_name = input_name
        self.condition = condition
        parts = []
        if input_name:
            parts.append("輸入「%s」" % input_name)
        if condition:
            parts.append("條件不成立：%s" % condition)
        parts.append(message)
        super().__init__("；".join(parts))
