#!/usr/bin/env python3
"""Phase 7: Page Quality Review (Optional)
Usage: python scripts/p7_review_pages.py output/mybook"""

import argparse, json, sys
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from utils import log, ProjectPaths, ensure_dir

FLAGS = {"great": {"color":"#22c55e","label":"Great","key":"1"}, "acceptable": {"color":"#3b82f6","label":"Acceptable","key":"2"},
         "poor": {"color":"#f59e0b","label":"Poor","key":"3"}, "crop_issue": {"color":"#ef4444","label":"Crop issue","key":"4"}}

class PageReviewApp:
    def __init__(self, root, output_dir):
        self.root = root; self.root.title("Page Quality Review"); self.root.configure(bg="#0a0a0a"); self.root.geometry("900x800")
        self.paths = ProjectPaths(output_dir); self.review_dir = ensure_dir(self.paths.reports)
        self.pages = json.loads((self.paths.json / "pages.json").read_text())
        self.current_idx = 0; self.flags = {}; self.notes = {}; self.photo = None
        existing = self.paths.json / "page_review.json"
        if existing.exists():
            try:
                d = json.loads(existing.read_text())
                self.flags = {int(k):v for k,v in d.get("flags",{}).items()}
                self.notes = {int(k):v for k,v in d.get("notes",{}).items()}
            except: pass
        self._build_ui(); self._bind_keys(); self._show_current()

    @staticmethod
    def _button(parent, command, **kw):
        # macOS Aqua tk.Button ignores bg/fg, so use a clickable Label instead.
        kw.setdefault("cursor", "hand2")
        lbl = tk.Label(parent, **kw)
        lbl.bind("<Button-1>", lambda e: command())
        return lbl

    def _build_ui(self):
        bg,fg,dim="#0a0a0a","#e2e8f0","#64748b"
        top=tk.Frame(self.root,bg="#111",height=40);top.pack(fill="x");top.pack_propagate(False)
        tk.Label(top,text="Page Review",font=("Menlo",13,"bold"),bg="#111",fg=fg).pack(side="left",padx=12)
        self.lbl_counter=tk.Label(top,text="",font=("Menlo",11),bg="#111",fg=dim);self.lbl_counter.pack(side="left",padx=8)
        self._button(top,self._save,text="Save (⌘S)",font=("Menlo",10),bg="#3b82f6",fg="white",relief="flat",padx=10,pady=4
                  ).pack(side="right",padx=8,pady=6)
        main=tk.Frame(self.root,bg=bg);main.pack(fill="both",expand=True)
        self.lbl_info=tk.Label(main,text="",font=("Menlo",11),bg=bg,fg=dim);self.lbl_info.pack(pady=(8,4))
        self.canvas=tk.Canvas(main,bg="#111",highlightthickness=0);self.canvas.pack(fill="both",expand=True,padx=12,pady=4)
        self.canvas.bind("<Configure>",lambda e:self._show_current())
        nav=tk.Frame(main,bg=bg);nav.pack(pady=(0,8))
        self._button(nav,self._prev,text="← Prev",font=("Menlo",11),bg="#1e293b",fg=fg,relief="flat",padx=16,pady=4).pack(side="left",padx=4)
        self._button(nav,self._next,text="Next →",font=("Menlo",11),bg="#1e293b",fg=fg,relief="flat",padx=16,pady=4).pack(side="left",padx=4)
        self.note_entry=tk.Text(main,font=("Menlo",10),bg="#111",fg=fg,insertbackground=fg,relief="flat",height=2,wrap="word")
        self.note_entry.pack(fill="x",padx=12,pady=4)

    def _bind_keys(self):
        self.root.bind("<Right>",lambda e:self._next());self.root.bind("<Left>",lambda e:self._prev())
        self.root.bind("d",lambda e:self._next());self.root.bind("a",lambda e:self._prev())
        for n,f in [("1","great"),("2","acceptable"),("3","poor"),("4","crop_issue")]:
            self.root.bind(n,lambda e,f=f:self._flag(f))
        self.root.bind("0",lambda e:self._clear_flag())
        self.root.bind("<Command-s>",lambda e:self._save())

    def _prev(self):
        self._save_note()
        if self.current_idx>0:self.current_idx-=1;self._show_current()
    def _next(self):
        self._save_note()
        if self.current_idx<len(self.pages)-1:self.current_idx+=1;self._show_current()

    def _show_current(self):
        pg=self.pages[self.current_idx];pn=pg["page_num"]
        self.lbl_info.config(text=f"Page {pn} | {pg['type']} | {pg['filename']}")
        self.lbl_counter.config(text=f"{self.current_idx+1}/{len(self.pages)}")
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        if cw<10:return
        try:
            img=Image.open(self.paths.pages/pg["filename"])
            iw,ih=img.size;scale=min(cw/iw,ch/ih,1.0)
            img=img.resize((int(iw*scale),int(ih*scale)),Image.LANCZOS)
            self.photo=ImageTk.PhotoImage(img);self.canvas.delete("all")
            self.canvas.create_image(cw//2,ch//2,image=self.photo,anchor="center")
        except:pass
        self.note_entry.delete("1.0","end")
        n=self.notes.get(pn,"")
        if n:self.note_entry.insert("1.0",n)

    def _flag(self,f):
        pn=self.pages[self.current_idx]["page_num"]
        if self.flags.get(pn)==f:del self.flags[pn]
        else:self.flags[pn]=f
        self._auto_save();self._show_current()
    def _clear_flag(self):
        self.flags.pop(self.pages[self.current_idx]["page_num"],None);self._auto_save();self._show_current()
    def _save_note(self):
        pn=self.pages[self.current_idx]["page_num"]
        t=self.note_entry.get("1.0","end").strip()
        if t:self.notes[pn]=t
        else:self.notes.pop(pn,None)
        self._auto_save()
    def _auto_save(self):
        d={"flags":{str(k):v for k,v in self.flags.items()},"notes":{str(k):v for k,v in self.notes.items()}}
        (self.paths.json/"page_review.json").write_text(json.dumps(d,indent=2))
    def _save(self):
        self._save_note()
        lines=["# Page Quality Review",f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               f"Total: {len(self.pages)}, Flagged: {len(self.flags)}, Notes: {len(self.notes)}",""]
        for key,cfg in FLAGS.items():
            items=[(pg,self.notes.get(pg["page_num"],"")) for pg in self.pages if self.flags.get(pg["page_num"])==key]
            if items:
                lines.append(f"## {cfg['label']} ({len(items)})")
                for pg,note in items:
                    ns=f' — "{note}"' if note else ""
                    lines.append(f"- Page {pg['page_num']} | {pg['filename']}{ns}")
                lines.append("")
        (self.paths.reports/"page_review_report.md").write_text("\n".join(lines))
        messagebox.showinfo("Saved",f"Flagged: {len(self.flags)}, Notes: {len(self.notes)}")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("output_dir")
    args=parser.parse_args()
    root=tk.Tk();PageReviewApp(root,args.output_dir);root.mainloop()

if __name__=="__main__":main()
