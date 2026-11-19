# -*- coding: utf-8 -*-
"""User-interface strings of the review app in English (default) and Chinese.

Language: --lang en|zh on the command line, else the RCF_LANG environment variable, else English.
The app's language button switches at run time.
"""
import os

LANGS = ("en", "zh")
_lang = "en"

STRINGS = {
    "rel.same_room": ("Same room", "同房"),
    "rel.different_room": ("Different room", "不同房"),
    "rel.uncertain": ("Uncertain", "不確定"),
    "rel.none": ("not judged", "未判定"),
    "title": ("Same-room candidate review - %s", "同房候選審查 — %s"),
    "help": (
        "How to use: (1) Pick a house at the top left.  (2) Set the thresholds: wedge slider to the left = more candidates "
        "(more same-room pairs found, but more wrong ones), to the right = more reliable; pairs below the thresholds are hidden. "
        "With \"Door-aware wedge\" checked, inliers seen through a door opening do not count toward the wedge (openings shown in cyan).  "
        "(3) Click a pair in the list on the left; panoramas A and B appear on the right, one above the other.  "
        "(4) Judge: S = same room, D = different room, U = uncertain, C = clear; the next pair is selected automatically "
        "(Up/Down to move). Click a panorama (or press A, B) to enlarge it, I = toggle the inlier marks, drag the divider "
        "between the two panoramas to resize them.  (5) Same-room judgements are chained into the room groups at the bottom left "
        "(A-B and B-C same room -> A, B, C one room); red text = conflict.  (6) Judgements are saved immediately; when done, "
        "press \"Export room groups JSON\" at the top right.",
        "操作方式：① 左上選「房子」　② 調整門檻：楔形往左拉＝候選變多（較易找到同房，但誤判也多），往右拉＝較可靠；"
        "低於門檻的配對不顯示；勾選「去門楔形」時，隔著門洞看到的內點不計入楔形（門洞以青色標示）　③ 點左邊清單的一對，右邊上下顯示全景 A、B　④ 判斷：S＝同房、D＝不同房、U＝不確定、"
        "C＝清除，按完自動跳下一對（↑↓ 可移動）；點全景（或按 A、B）可放大，I＝開關內點標示，拖曳兩張之間的分隔線可調整上下大小　⑤ 同房判定會自動串成左下方的「房間群組」（A–B、B–C 同房 → A、B、C 同一間）；"
        "紅字為矛盾　⑥ 判定即時存檔，看完按右上「匯出房間群組 JSON」。"),
    "lang_button": ("中文", "English"),
    "house": ("House", "房子"),
    "open_chk": ("Door-aware wedge (SAM3 openings)", "去門楔形（SAM3 門洞）"),
    "wedge_lbl": ("Both wedges >=", "兩側楔形 ≥"),
    "inl_lbl": ("deg   inliers >=", "°   內點 ≥"),
    "hide": ("Hide judged", "隱藏已判定"),
    "show_inl": ("Show inliers (I)", "標示內點 (I)"),
    "export": ("Export room groups JSON", "匯出房間群組 JSON"),
    "cols": (("Panorama A", "Panorama B", "Min wedge", "Inliers", "Door %", "Decision"),
             ("全景 A", "全景 B", "較窄楔形°", "內點", "門洞內點", "判定")),
    "groups_lbl": ("Room groups (chained from \"same room\" judgements)", "房間群組（由「同房」判定串成）"),
    "btn.same_room": ("Same room (S)", "同房 (S)"),
    "btn.different_room": ("Different room (D)", "不同房 (D)"),
    "btn.uncertain": ("Uncertain (U)", "不確定 (U)"),
    "btn.clear": ("Clear (C)", "清除 (C)"),
    "prev": ("Previous (Up)", "上一組 (↑)"),
    "next": ("Next (Down)", "下一組 (↓)"),
    "mode.open": ("door-aware wedge", "去門楔形"),
    "mode.plain": ("wedge", "楔形"),
    "expect": ("Dollhouse calibration (%s): expected P=%.2f, R=%.2f at this threshold",
               "Dollhouse 校準（%s）：此門檻預期 P≈%.2f、R≈%.2f"),
    "status": ("%d candidate pairs, %d judged | %d panoramas without candidates (treated as single-panorama rooms) | pairs scanned %d / %d",
               "候選 %d 對，已判定 %d 對 ｜ 沒有候選的全景 %d 張（視為單張房間）｜ 已掃描配對 %d / %d"),
    "ov.none": ("(no GeoVerify detail file for this pair; inliers cannot be shown)",
                "（這一對沒有 GeoVerify 明細檔，無法標示內點）"),
    "ov.points": ("Shown: %d inliers (same colour = matching points in A and B) + %d co-visible regions (R1..., box = extent of that region's inliers)",
                  "標示：%d 個內點（同色＝A、B 中的對應點）＋ %d 塊共視區域（R1…，框＝該群內點的範圍）"),
    "ov.door": ("; cyan = SAM3 door opening, grey hollow dots = inliers seen through an opening (not counted in the door-aware wedge)",
                "；青色＝SAM3 門洞，灰色空心點＝隔著門洞的內點（不計入去門楔形）"),
    "ov.regions": ("Shown: %d co-visible regions (R1...). This scan has no per-point inliers; run tools\\recompute_inliers.py to add them",
                   "標示：%d 塊共視區域（R1…）。這份掃描沒有存逐點內點，執行 tools\\recompute_inliers.py 可補上"),
    "zoom_title": ("%s: %s (Esc or click to close)", "%s：%s（Esc 或點一下關閉）"),
    "img_err": ("Cannot read the image: %s", "無法讀取影像：%s"),
    "side_open": ("%.0f deg (door-aware %.0f deg)", "%.0f°（去門 %.0f°）"),
    "side": ("%.0f deg", "%.0f°"),
    "cap_a": ("A: %s   (A-side wedge %s)", "A：%s　（A 側楔形 %s）"),
    "cap_b": ("B: %s   (B-side wedge %s)", "B：%s　（B 側楔形 %s）"),
    "extra": ("; door-aware narrower wedge %.0f deg, remaining inliers %d, door inliers %.0f%%",
              "；去門後較窄楔形 %.0f°、剩餘內點 %d、門洞內點 %.0f%%"),
    "info": ("Pair %d / %d   narrower wedge %.0f deg, E inliers %d, inlier rate %s, co-visible regions %d%s   current decision: %s",
             "第 %d / %d 組　較窄楔形 %.0f°、E 內點 %d、內點率 %s、共視區域 %d 塊%s　目前判定：%s"),
    "empty_tip": ("uncheck \"Door-aware wedge\", or ", "取消勾選「去門楔形」，或"),
    "empty": ("No candidate pairs in this house at the current thresholds (every panorama is treated as a single-panorama room).\n"
              "You can %smove the \"Both wedges\" slider to the left, or pick another house at the top left.",
              "這棟房子在目前門檻下沒有候選配對（全部全景視為單張房間）。\n"
              "可以%s把「兩側楔形」滑桿往左調，或從左上角換一棟房子。"),
    "no_cand": ("No candidate pairs at the current thresholds", "目前門檻下沒有候選配對"),
    "no_groups": ("No groups yet.\n", "尚無群組。\n"),
    "group_line": ("Room %d (%d panoramas): %s\n", "房間 %d（%d 張）：%s\n"),
    "list_sep": (", ", "、"),
    "conflict": ("Conflict: %s and %s were judged different rooms but end up in the same group through other same-room judgements\n",
                 "矛盾：%s 與 %s 被判不同房，卻經由其他同房判定落在同一群組\n"),
    "exported": ("Exported %d room groups.", "已匯出 %d 個房間群組。"),
    "exported_conf": ("\nNote: %d conflicts (groups that contain a pair judged different rooms); written to \"conflicts\".",
                      "\n注意：有 %d 個矛盾（群組內含被判不同房的配對），已寫入 conflicts。"),
    "export_title": ("Export complete", "匯出完成"),
    "save_title": ("Save room groups JSON", "儲存房間群組 JSON"),
    "pick_title": ("Choose a scan folder (with manifest.json)", "選擇掃描結果資料夾（含 manifest.json）"),
    "notfound_title": ("Scan not found", "找不到掃描結果"),
    "notfound": ("%s has no manifest.json. Run the scan first (Import_Dataset.ps1).", "%s 裡沒有 manifest.json。請先執行掃描（Import_Dataset.ps1）。"),
}


def set_lang(lang):
    global _lang
    _lang = lang if lang in LANGS else "en"


def get_lang():
    return _lang


def T(key):
    return STRINGS[key][LANGS.index(_lang)]


def rel_text(relation):
    return T("rel." + relation) if relation else ""


set_lang(os.environ.get("RCF_LANG", "en").strip().lower()[:2])
