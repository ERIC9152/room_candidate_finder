# -*- coding: utf-8 -*-
"""Desktop review app.

    python -m finder.app [--scan scans\\dollhouse] [--reviewer NAME]

Without --scan a folder picker opens in scans/.
Keys: S same room, D different room, U uncertain, C clear, Down/N next, Up/P previous.
Language: English by default; --lang zh, the RCF_LANG environment variable, or the language button switch to Chinese.
Decisions are saved immediately to <scan>/decisions.jsonl; reopen to continue.
"""
import argparse
import colorsys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from . import config
from .i18n import T, get_lang, rel_text, set_lang
from .model import ReviewModel


class App:
    def __init__(self, root, model, max_w=980):
        self.root, self.m, self.max_w = root, model, max_w
        houses = sorted(model.houses)
        # start on the first house that has candidates under the default rule (an empty first house
        # would leave the panorama area blank)
        first = next((h for h in houses if model.candidates(h)), houses[0] if houses else "")
        self.house = tk.StringVar(value=first)
        self.empty_msg = ""
        self.wedge = tk.DoubleVar(value=model.wedge_deg)
        self.inl = tk.IntVar(value=model.min_inliers)
        self.hide = tk.BooleanVar(value=False)
        self.open_mode = tk.BooleanVar(value=model.mode == "open")
        self.rows, self.photos, self._last_index = [], [], 0
        root.title(T("title") % os.path.basename(model.scan_dir))
        root.geometry("%dx%d+0+0" % (min(1500, root.winfo_screenwidth() - 20), min(980, root.winfo_screenheight() - 80)))
        try:
            root.state("zoomed")              # Windows: start maximised so both panoramas get room
        except tk.TclError:
            pass
        self.src, self._rz, self.detail, self.orig = [None, None], None, None, [None, None]
        self.omask = [None, None]                 # SAM3 opening masks of A and B (PIL "L", 2048x1024) or None
        self.show_inl = tk.BooleanVar(value=True)
        self._build()
        self.refresh()

    # ---------- layout
    # (help text: i18n.T("help"))

    def toggle_lang(self):
        """Switch English <-> Chinese and rebuild the window (decisions are already on disk)."""
        set_lang("zh" if get_lang() == "en" else "en")
        sel = self.tree.selection()
        for w in self.root.winfo_children():
            w.destroy()
        self.root.title(T("title") % os.path.basename(self.m.scan_dir))
        self._build()
        self.refresh()
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0]); self.tree.see(sel[0])

    def _build(self):
        bar = tk.Frame(self.root, bg="#eef3f8", padx=10, pady=6); bar.pack(fill="x")
        self.help_lbl = tk.Label(bar, text=T("help"), bg="#eef3f8", fg="#1b2430", justify="left", anchor="w",
                                 font=("Microsoft JhengHei UI", 10))
        self.help_lbl.pack(fill="x")
        bar.bind("<Configure>", lambda e: self.help_lbl.config(wraplength=max(400, e.width - 24)))
        top = ttk.Frame(self.root, padding=8); top.pack(fill="x")
        # right-hand buttons first: pack() gives space in packing order, so they are never squeezed out
        ttk.Button(top, text=T("export"), command=self.export).pack(side="right")
        ttk.Button(top, text=T("lang_button"), command=self.toggle_lang).pack(side="right", padx=6)
        ttk.Label(top, text=T("house")).pack(side="left")
        cb = ttk.Combobox(top, textvariable=self.house, values=sorted(self.m.houses), width=34, state="readonly")
        cb.pack(side="left", padx=(4, 16)); cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.open_chk = ttk.Checkbutton(top, text=T("open_chk"), variable=self.open_mode, command=self._mode_changed)
        self.open_chk.pack(side="left", padx=(0, 12))
        if not self.m.has_open:
            self.open_chk.state(["disabled"])
        ttk.Label(top, text=T("wedge_lbl")).pack(side="left")
        tk.Scale(top, from_=0, to=90, resolution=5, orient="horizontal", length=220, variable=self.wedge,
                 command=lambda v: self._thresholds_changed()).pack(side="left")
        ttk.Label(top, text=T("inl_lbl")).pack(side="left")
        sp = ttk.Spinbox(top, from_=8, to=1000, increment=10, width=6, textvariable=self.inl,
                         command=self._thresholds_changed)
        sp.pack(side="left", padx=(4, 12)); sp.bind("<Return>", lambda e: self._thresholds_changed())
        top2 = ttk.Frame(self.root, padding=(8, 0, 8, 6)); top2.pack(fill="x")      # second row: view options + calibration
        ttk.Checkbutton(top2, text=T("hide"), variable=self.hide, command=self.refresh).pack(side="left")
        ttk.Checkbutton(top2, text=T("show_inl"), variable=self.show_inl, command=self._render).pack(side="left", padx=16)
        self.expect = ttk.Label(top2, text="", foreground="#2f5d8a"); self.expect.pack(side="left", padx=12)

        body = ttk.Panedwindow(self.root, orient="horizontal"); body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=(8, 0)); body.add(left, weight=1)
        cols = ("a", "b", "wedge", "inl", "door", "rel")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=12, selectmode="browse")
        for c, t, w in zip(cols, T("cols"), (130, 130, 75, 55, 60, 85)):
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="w" if c in ("a", "b") else "e")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.show_current())
        self.info = ttk.Label(left, text="", justify="left", wraplength=480); self.info.pack(fill="x", pady=6)
        ttk.Label(left, text=T("groups_lbl")).pack(anchor="w")
        self.groups_txt = tk.Text(left, height=8, wrap="word"); self.groups_txt.pack(fill="both", expand=False)
        self.groups_txt.tag_configure("bad", foreground="#b23b3b")

        right = ttk.Frame(body, padding=(8, 0)); body.add(right, weight=3)
        btns = ttk.Frame(right, padding=(0, 0, 0, 6)); btns.pack(side="top")
        # A above B; each panorama is scaled to fit its pane (whole image always visible).
        # Drag the divider between them to give one more room; click an image to open it large.
        panes = ttk.Panedwindow(right, orient="vertical"); panes.pack(fill="both", expand=True)
        self.cap, self.canv = [], []
        for k in range(2):
            f = ttk.Frame(panes); panes.add(f, weight=1)
            cap = ttk.Label(f, text=""); cap.pack(anchor="w")
            cv = tk.Canvas(f, highlightthickness=0, bg="#20262e", width=200, height=100, cursor="hand2")
            cv.pack(fill="both", expand=True)
            cv.bind("<Configure>", lambda e: self._schedule_render())
            cv.bind("<Button-1>", lambda e, k=k: self.zoom(k))
            self.cap.append(cap); self.canv.append(cv)
        self.cap_a, self.cap_b = self.cap
        # give the panoramas most of the width (the divider between list and images can still be dragged)
        def first_layout(e, done=[]):
            if not done and e.width > 50:
                done.append(1)
                self.root.after_idle(lambda: body.sashpos(0, max(420, int(e.width * .34))))
        body.bind("<Configure>", first_layout, add="+")
        for text, rel in ((T("btn.same_room"), "same_room"), (T("btn.different_room"), "different_room"), (T("btn.uncertain"), "uncertain"), (T("btn.clear"), None)):
            ttk.Button(btns, text=text, command=lambda r=rel: self.decide(r)).pack(side="left", padx=6)
        ttk.Button(btns, text=T("prev"), command=lambda: self.move(-1)).pack(side="left", padx=(24, 6))
        ttk.Button(btns, text=T("next"), command=lambda: self.move(1)).pack(side="left", padx=6)
        self.status = ttk.Label(self.root, text="", padding=6, relief="sunken")
        self.status.pack(fill="x", side="bottom", before=body)    # packed before the body so it is never squeezed out
        for key, rel in (("s", "same_room"), ("d", "different_room"), ("u", "uncertain"), ("c", None)):
            self.root.bind(key, lambda e, r=rel: self.decide(r)); self.root.bind(key.upper(), lambda e, r=rel: self.decide(r))
        self.root.bind("<Down>", lambda e: self.move(1)); self.root.bind("n", lambda e: self.move(1))
        self.root.bind("<Up>", lambda e: self.move(-1)); self.root.bind("p", lambda e: self.move(-1))
        self.root.bind("a", lambda e: self.zoom(0)); self.root.bind("b", lambda e: self.zoom(1))
        self.root.bind("i", lambda e: (self.show_inl.set(not self.show_inl.get()), self._render()))

    # ---------- data -> view
    def _mode_changed(self):
        """Switch plain / door-aware wedge; the slider jumps to that mode's default threshold."""
        self.m.set_mode("open" if self.open_mode.get() else "plain")
        self.open_mode.set(self.m.mode == "open")
        self.wedge.set(self.m.wedge_deg)
        self.refresh()

    def _thresholds_changed(self):
        try:
            self.m.wedge_deg = float(self.wedge.get()); self.m.min_inliers = int(self.inl.get())
        except (tk.TclError, ValueError):
            return
        self.refresh()

    def refresh(self, keep=None):
        h = self.house.get()
        self.rows = self.m.candidates(h, hide_decided=self.hide.get()) if h else []
        self.tree.delete(*self.tree.get_children())
        for i, c in enumerate(self.rows):
            s = c["score"]
            o = s.get("open")
            door = "%.0f%%" % (100 * o["door_frac"]) if o else "—"
            self.tree.insert("", "end", iid=str(i), values=(c["a"], c["b"], "%.0f" % c["wedge"], s["n_inlier"], door, rel_text(c["relation"])))
        q = self.m.expected_quality()
        self.expect.config(text=(T("expect")
                                 % (T("mode.open") if self.m.mode == "open" else T("mode.plain"), q["P"], q["R"])) if q else "")
        pr = self.m.progress(h) if h else dict(candidates=0, decided=0)
        none = self.m.panos_without_candidates(h) if h else []
        scanned, expected = (self.m.scanned_pairs(h), self.m.expected_pairs(h)) if h else (0, 0)
        self.status.config(text=T("status")
                           % (pr["candidates"], pr["decided"], len(none), scanned, expected))
        self.render_groups()
        if self.rows:
            idx = 0
            if keep is not None:
                for i, c in enumerate(self.rows):
                    if (c["a"], c["b"]) == keep:
                        idx = i; break
                else:
                    idx = min(self._last_index, len(self.rows) - 1)
            self.tree.selection_set(str(idx)); self.tree.see(str(idx))
        else:
            self.clear_images()

    def current(self):
        sel = self.tree.selection()
        return (int(sel[0]), self.rows[int(sel[0])]) if sel else (None, None)

    def _load(self, path, w=2400, k=None):
        with Image.open(path) as im:
            if k is not None:
                self.orig[k] = im.size                  # full-resolution size (GeoVerify pixel coordinates)
            im.draft("RGB", (w, w // 2))          # fast JPEG decode at reduced size
            im = im.convert("RGB")
        im.thumbnail((w, w // 2))
        return im

    def _schedule_render(self):
        if self._rz:
            self.root.after_cancel(self._rz)
        self._rz = self.root.after(60, self._render)

    def _render(self):
        """Fit each panorama into its canvas (keeps 2:1, never cropped)."""
        self._rz, self.photos = None, []
        for im, cv in zip(self.src, self.canv):
            cv.delete("all")
            if im is None:
                if self.empty_msg and cv is self.canv[0]:
                    cv.create_text(cv.winfo_width() // 2, cv.winfo_height() // 2, text=self.empty_msg, fill="#e8eef5",
                                   font=("Microsoft JhengHei UI", 13), justify="center", width=max(200, cv.winfo_width() - 60))
                continue
            w, h = cv.winfo_width(), cv.winfo_height()
            s = min(w / im.width, h / im.height)
            if s <= 0:
                continue
            k = self.canv.index(cv)
            disp = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
            if self.show_inl.get():
                self._overlay(disp, k)
            ph = ImageTk.PhotoImage(disp)
            self.photos.append(ph)
            cv.create_image(w // 2, h // 2, image=ph, anchor="center")

    PAL = [(255, 64, 64), (64, 160, 255), (255, 200, 0), (0, 210, 120), (220, 90, 255), (255, 140, 40)]

    def _overlay(self, img, k, big=False):
        """Draw GeoVerify's inliers (same colour = same correspondence in A and B; colour follows the
        position in A) and covisibility regions (convex hull of each inlier cluster) onto img."""
        g = self.detail
        if not g:
            return
        size = (g.get("img_size_a") if k == 0 else g.get("img_size_b")) or self.orig[k]
        if not size:
            return
        f = img.width / float(size[0])
        show_open = self.m.mode == "open" and any(m is not None for m in self.omask)
        if show_open and self.omask[k] is not None:          # tint the openings of this panorama
            m = self.omask[k].resize(img.size, Image.NEAREST).point(lambda v: 110 if v else 0)
            img.paste(Image.new("RGB", img.size, (0, 230, 255)), (0, 0), m)
        d = ImageDraw.Draw(img)
        pts = g.get("inliers") or []
        if pts:
            step = max(1, len(pts) // (900 if big else 400))
            r = 3 if big else 2
            wa = float((g.get("img_size_a") or self.orig[0])[0])
            for p in pts[::step]:
                x, y = (p[0], p[1]) if k == 0 else (p[2], p[3])
                x, y = x * f, y * f
                if show_open and self._through_opening(p, g):
                    # seen through an opening in A or B: not counted by the door-aware wedge
                    d.ellipse([x - r, y - r, x + r, y + r], fill=None, outline=(170, 170, 170))
                    continue
                rgb = tuple(int(255 * c) for c in colorsys.hsv_to_rgb((p[0] / wa) % 1.0, .9, 1.))
                d.ellipse([x - r, y - r, x + r, y + r], fill=rgb, outline=(0, 0, 0))
        for i, reg in enumerate(g.get("regions") or []):
            side = reg.get("a" if k == 0 else "b") or {}
            poly = [(x * f, y * f) for x, y in side.get("polygon") or []]
            if len(poly) >= 3:
                col = self.PAL[i % len(self.PAL)]
                d.line(poly + poly[:1], fill=col, width=3 if big else 2)
                x0, y0 = min(p[0] for p in poly), min(p[1] for p in poly)
                d.rectangle([x0, y0 - 14, x0 + 70, y0], fill=(0, 0, 0))
                d.text((x0 + 3, y0 - 13), "R%d n=%d" % (i + 1, reg.get("n", 0)), fill=col)

    def _through_opening(self, p, g):
        """Is this inlier inside the opening mask of A or of B (display check, mask not dilated)?"""
        for m, xi, yi, sz in ((self.omask[0], 0, 1, g.get("img_size_a") or self.orig[0]),
                              (self.omask[1], 2, 3, g.get("img_size_b") or self.orig[1])):
            if m is None or not sz:
                continue
            x = min(m.width - 1, max(0, int(p[xi] / float(sz[0]) * m.width)))
            y = min(m.height - 1, max(0, int(p[yi] / float(sz[1]) * m.height)))
            if m.getpixel((x, y)):
                return True
        return False

    def _load_mask(self, house, pid):
        fp = self.m.opening_mask_path(house, pid)
        if not fp:
            return None
        with Image.open(fp) as im:
            return im.convert("L").resize((2048, 1024), Image.NEAREST)

    def _overlay_text(self):
        g = self.detail
        if g is None:
            return T("ov.none")
        n_pts, n_reg = len(g.get("inliers") or []), len(g.get("regions") or [])
        if n_pts:
            t = T("ov.points") % (n_pts, n_reg)
            if self.m.mode == "open" and any(m is not None for m in self.omask):
                t += T("ov.door")
            return t
        return T("ov.regions") % n_reg

    def zoom(self, k):
        """Open panorama A (k=0) or B (k=1) as large as the screen allows; Esc or click closes."""
        i, c = self.current()
        if c is None:
            return
        name = c["a"] if k == 0 else c["b"]
        im = self._load(self.m.image_path(self.house.get(), name), w=4096)
        sw, sh = self.root.winfo_screenwidth() - 40, self.root.winfo_screenheight() - 120
        s = min(sw / im.width, sh / im.height)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        if self.show_inl.get():
            self._overlay(im, k, big=True)
        top = tk.Toplevel(self.root); top.title(T("zoom_title") % ("AB"[k], name))
        ph = ImageTk.PhotoImage(im)
        lbl = tk.Label(top, image=ph, cursor="hand2"); lbl.image = ph; lbl.pack()
        for w in (top, lbl):
            w.bind("<Escape>", lambda e: top.destroy()); w.bind("<Button-1>", lambda e: top.destroy())
        top.geometry("+10+10"); top.focus_set()

    def show_current(self):
        i, c = self.current()
        if c is None:
            return
        self._last_index = i
        self.empty_msg = ""
        h = self.house.get(); s = c["score"]
        try:
            self.src = [self._load(self.m.image_path(h, c["a"]), k=0), self._load(self.m.image_path(h, c["b"]), k=1)]
            self.detail = self.m.pair_detail(h, c["a"], c["b"])
            self.omask = [self._load_mask(h, c["a"]), self._load_mask(h, c["b"])]
        except OSError as e:
            self.info.config(text=T("img_err") % e); return
        self._render()
        o = s.get("open")
        side = lambda w, ow: (T("side_open") % (w, ow)) if o else T("side") % w
        self.cap_a.config(text=T("cap_a") % (c["a"], side(s["wedge_a"], o["wedge_a"] if o else 0)))
        self.cap_b.config(text=T("cap_b") % (c["b"], side(s["wedge_b"], o["wedge_b"] if o else 0)))
        extra = (T("extra") % (o["wedge_min"], o["n_keep"], 100 * o["door_frac"])) if o else ""
        self.info.config(text=T("info")
                         % (i + 1, len(self.rows), s["wedge_min"], s["n_inlier"],
                            "%.2f" % s["inlier_rate"] if s.get("inlier_rate") is not None else "—",
                            s["n_regions"], extra, rel_text(c["relation"]) or T("rel.none")) + "\n" + self._overlay_text())

    def clear_images(self):
        self.src, self.photos, self.detail, self.omask = [None, None], [], None, [None, None]
        tip = T("empty_tip") if self.m.mode == "open" else ""
        self.empty_msg = (T("empty") % tip) if self.house.get() else ""
        self._render()
        self.cap_a.config(text=T("no_cand")); self.cap_b.config(text=""); self.info.config(text="")

    def render_groups(self):
        h = self.house.get()
        self.groups_txt.delete("1.0", "end")
        if not h:
            return
        groups, conflicts = self.m.groups(h)
        if not groups:
            self.groups_txt.insert("end", T("no_groups"))
        for i, g in enumerate(groups, 1):
            self.groups_txt.insert("end", T("group_line") % (i, len(g), T("list_sep").join(g)))
        for a, b in conflicts:
            self.groups_txt.insert("end", T("conflict") % (a, b), "bad")

    # ---------- actions
    def decide(self, rel):
        i, c = self.current()
        if c is None:
            return
        self.m.decide(self.house.get(), c["a"], c["b"], rel)
        nxt = self.rows[i + 1] if i + 1 < len(self.rows) else None
        self.refresh(keep=(nxt["a"], nxt["b"]) if (nxt and rel is not None) else (c["a"], c["b"]))

    def move(self, step):
        i, c = self.current()
        if c is None:
            return
        j = max(0, min(len(self.rows) - 1, i + step))
        self.tree.selection_set(str(j)); self.tree.see(str(j))

    def export(self):
        path = filedialog.asksaveasfilename(title=T("save_title"), initialdir=self.m.scan_dir, initialfile="room_groups.json",
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        data = self.m.export(path)
        msg = T("exported") % len(data["groups"])
        if data["conflicts"]:
            msg += T("exported_conf") % len(data["conflicts"])
        messagebox.showinfo(T("export_title"), msg)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Review same-room candidates")
    ap.add_argument("--scan", default="", help="output folder of finder.scan (asks when omitted)")
    ap.add_argument("--reviewer", default=os.environ.get("USERNAME", ""))
    ap.add_argument("--lang", choices=("en", "zh"), default=None, help="interface language (default: RCF_LANG or en)")
    a = ap.parse_args(argv)
    if a.lang:
        set_lang(a.lang)
    root = tk.Tk()
    scan = a.scan
    if not scan:
        os.makedirs(config.SCANS_DIR, exist_ok=True)
        root.withdraw()
        scan = filedialog.askdirectory(title=T("pick_title"), initialdir=config.SCANS_DIR)
        root.deiconify()
        if not scan:
            return
    if not os.path.exists(os.path.join(scan, "manifest.json")):
        messagebox.showerror(T("notfound_title"), T("notfound") % scan)
        return
    App(root, ReviewModel(scan, a.reviewer))
    root.mainloop()


if __name__ == "__main__":
    main()
