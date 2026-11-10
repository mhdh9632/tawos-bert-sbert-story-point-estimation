"""Build the final NLP project report PDF from verified experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/final_report.pdf"
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#1F6E8C")
PALE = colors.HexColor("#EAF4F8")
GOLD = colors.HexColor("#D89B2B")
TEXT = colors.HexColor("#273444")
MUTED = colors.HexColor("#66788A")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, *, header=True, font_size=8.3, alignments=None):
    result = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7D1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if alignments:
        for column, alignment in enumerate(alignments):
            commands.append(("ALIGN", (column, 1), (column, -1), alignment))
    result.setStyle(TableStyle(commands))
    return result


def draw_page(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D7E2E8"))
    canvas.line(1.7 * cm, height - 1.25 * cm, width - 1.7 * cm, height - 1.25 * cm)
    canvas.setFont("Helvetica", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.7 * cm, height - 0.95 * cm, "NLP Project — Problem Statement 10")
    canvas.drawRightString(width - 1.7 * cm, 0.75 * cm, f"Page {document.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    validation = read_csv(ROOT / "results/DM_validation_summary.csv")
    final = read_csv(ROOT / "results/DM_final_test_metrics.csv")
    folds = read_csv(ROOT / "results/DM_fold_metrics.csv")
    experiment = json.loads((ROOT / "results/DM_experiment_summary.json").read_text())

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.35 * cm,
        title="Combined BERT and SBERT Embeddings for Temporal Story-Point Estimation",
        author="[ADD GROUP MEMBER NAMES]",
        subject="Natural Language Processing Project — Problem Statement 10",
    )
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25,
            leading=31, textColor=NAVY, alignment=TA_LEFT, spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=13,
            leading=18, textColor=BLUE, spaceAfter=13,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
            leading=20, textColor=NAVY, spaceBefore=12, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
            leading=15, textColor=BLUE, spaceBefore=9, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4,
            leading=13.5, textColor=TEXT, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8.1,
            leading=11, textColor=MUTED, spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=11,
            leading=15, textColor=NAVY, leftIndent=12, rightIndent=12, spaceBefore=8, spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=13, textColor=TEXT, leftIndent=14, firstLineIndent=-7, bulletIndent=5, spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.8,
            leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=7,
        ),
    }

    story = []
    story.extend(
        [
            Spacer(1, 2.0 * cm),
            p("Natural Language Processing Project", styles["subtitle"]),
            p("Combined BERT and SBERT Embeddings for Temporal Story-Point Estimation", styles["title"]),
            p("Problem Statement 10 · Within-project · Temporal ordering", styles["subtitle"]),
            Spacer(1, 0.5 * cm),
            Table(
                [
                    [p("GROUP MEMBERS", styles["small"]), p("[ADD GROUP MEMBER NAMES AND IDs]", styles["body"])],
                    [p("GITHUB", styles["small"]), p('<link href="https://github.com/mhdh9632/tawos-bert-sbert-story-point-estimation">github.com/mhdh9632/tawos-bert-sbert-story-point-estimation</link>', styles["body"])],
                    [p("EXECUTED COLAB", styles["small"]), p('<link href="https://colab.research.google.com/drive/1IITpLs0qPoyH-MEnmVGMHTFEbK4Cu0yx?usp=sharing">Open completed experiment</link>', styles["body"])],
                    [p("DATE", styles["small"]), p("11 November 2026", styles["body"])],
                ],
                colWidths=[3.4 * cm, 12.6 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), PALE),
                        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#BCD0DA")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E2E8")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            Spacer(1, 1.0 * cm),
            Table(
                [[p("KEY RESULT", styles["small"])], [p("Combined BERT + SBERT achieved final-test MAE <b>3.9416</b> and MdAE <b>1.6988</b>, the lowest errors among all evaluated representations.", styles["callout"])]],
                colWidths=[16 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), GOLD),
                        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF8E8")),
                        ("BOX", (0, 0), (-1, -1), 0.8, GOLD),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    story += [p("Abstract", styles["h1"])]
    story += [p(
        "This project investigates whether combining contextual word-level information from BERT with sentence-level semantic information from SBERT improves software story-point estimation for later issues within the same project. TAWOS issues were filtered for usable text, dates, and story-point labels while excluding records changed after estimation. Lsstcorp Data Management (DM) supplied 14,782 eligible issues. The oldest 11,825 formed development data and the newest 2,957 formed the held-out test set. Five expanding-window validation folds preserved chronology. Frozen bert-base-uncased and all-MiniLM-L6-v2 encoders produced 768- and 384-dimensional vectors; concatenation produced 1,152 features. The combined model achieved final-test MAE 3.9416 and MdAE 1.6988, outperforming both baselines. Validation MdAE was mixed, so the result supports fusion for this target period without establishing universal superiority.", styles["body"])]

    story += [p("1. Introduction", styles["h1"])]
    story += [p(
        "Agile teams use story points to express relative issue effort or complexity. Estimation depends on technical detail, project vocabulary, and historical practice. NLP models can learn relationships between issue descriptions and previously assigned points, but evaluation must reflect deployment: past issues should predict future ones.", styles["body"])]
    story += [p(
        "Problem Statement 10 asks whether combined word and sentence embeddings effectively predict story points of later issues within the same project. The research question is: <b>Does concatenating BERT-derived contextual representations and SBERT sentence embeddings reduce MAE and MdAE on a chronological held-out period?</b>", styles["body"])]

    story += [p("2. Dataset", styles["h1"]), p("2.1 TAWOS and modelling fields", styles["h2"])]
    story += [p(
        "TAWOS V1.1 contains 458,232 issues from 39 open-source projects in 12 public Jira repositories. The model used Issue_Key, Title, Description_Text, Creation_Date, Story_Point, Project_ID, and the three changed-after-estimation flags. Resolution, time spent, total effort, comments, and other post-outcome fields were excluded to prevent leakage.", styles["body"])]
    story += [p("2.2 Eligibility and project selection", styles["h2"])]
    story += [p(
        "Eligible issues required usable title or description text, a valid timestamp, a finite non-negative label, a recognised project, and no indication that title, description, or story point changed after estimation. Title and description were joined with a [SEP] marker. A memory-bounded parser extracted only required SQL columns and cached a compact CSV in Drive.", styles["body"])]
    story += [p(
        "The notebook selected DM because it had the largest eligible sample: 14,782 issues dated 4 March 2014 to 22 October 2020. Median story points were 2.0 and the mean was 5.9843, indicating a right-skewed target distribution.", styles["body"])]
    image = Image(str(ROOT / "results/DM_label_distribution.png"), width=15.8 * cm, height=5.25 * cm)
    story += [image, p("Figure 1. Story-point distributions in development and held-out test data.", styles["caption"])]

    story += [p("3. Experimental Methodology", styles["h1"]), p("3.1 Temporal holdout", styles["h2"])]
    story += [p(
        "Issues were ordered by Creation_Date, with issue ID for stable tie ordering. The newest 20% was reserved before model selection. Equal timestamps were kept on the same side of a boundary. The final test set did not affect scaling, early stopping, or epoch selection.", styles["body"])]
    story += [table(
        [["Partition", "Issues", "Start", "End"],
         ["Development", "11,825", "2014-03-04 14:51", "2019-11-07 22:44"],
         ["Final test", "2,957", "2019-11-07 22:50", "2020-10-22 14:51"]],
        [3.2 * cm, 2.2 * cm, 5.2 * cm, 5.2 * cm], alignments=["LEFT", "RIGHT", "LEFT", "LEFT"]
    )]
    story += [p("3.2 Five expanding-window folds", styles["h2"])]
    split_rows = [["Fold", "Train", "Validation", "Train end", "Validation period"]]
    fold_periods = [
        ("1", "1,952", "1,955", "2016-02-03", "2016-02 to 2016-11"),
        ("2", "3,907", "1,962", "2016-11-30", "2016-11 to 2017-09"),
        ("3", "5,869", "1,993", "2017-09-28", "2017-09 to 2018-08"),
        ("4", "7,862", "1,994", "2018-08-06", "2018-08 to 2019-04"),
        ("5", "9,856", "1,969", "2019-04-08", "2019-04 to 2019-11"),
    ]
    split_rows.extend(fold_periods)
    story += [table(split_rows, [1.2 * cm, 2.4 * cm, 2.4 * cm, 4 * cm, 5.8 * cm], alignments=["CENTER", "RIGHT", "RIGHT", "LEFT", "LEFT"])]

    story += [p("3.3 Representations and prediction model", styles["h2"])]
    bullets = [
        "<b>BERT-only:</b> bert-base-uncased contextual token states, attention-mask-aware mean pooled to 768 dimensions.",
        "<b>SBERT-only:</b> all-MiniLM-L6-v2 semantic sentence vectors with 384 dimensions.",
        "<b>Combined:</b> concatenated BERT and SBERT vectors with 1,152 dimensions.",
        "<b>Regressor:</b> Linear(256) → ReLU → Dropout(0.20) → Linear(64) → ReLU → Dropout(0.10) → Linear(1).",
    ]
    for item in bullets:
        story.append(p("• " + item, styles["bullet"]))
    story += [p(
        "Encoders were frozen. Features and targets were standardised from training data only. AdamW used learning rate 0.001, weight decay 0.0001, batch size 64, Smooth L1 loss, maximum 50 epochs, patience 7, and seed 42. Negative predictions were clipped to zero. Median best epoch across folds set final training duration.", styles["body"])]

    story += [PageBreak(), p("4. Results", styles["h1"]), p("4.1 Five-fold validation summary", styles["h2"])]
    validation_data = [["Representation", "MAE mean", "MAE SD", "MdAE mean", "MdAE SD"]]
    for row in validation:
        validation_data.append([
            row["representation"].upper() if row["representation"] != "combined" else "Combined",
            f'{float(row["mae_mean"]):.4f}', f'{float(row["mae_std"]):.4f}',
            f'{float(row["mdae_mean"]):.4f}', f'{float(row["mdae_std"]):.4f}',
        ])
    story += [table(validation_data, [4.7 * cm, 2.8 * cm, 2.6 * cm, 2.8 * cm, 2.6 * cm], alignments=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])]
    story += [p(
        "Combined embeddings achieved the lowest mean validation MAE (4.5239), but SBERT achieved the lowest mean validation MdAE (2.0242). Fusion therefore did not uniformly dominate both baselines during model development.", styles["body"])]
    story += [p("4.2 Final held-out test", styles["h2"])]
    final_data = [["Representation", "MAE", "MdAE", "Epochs", "Dimensions"]]
    for row in sorted(final, key=lambda item: {"bert": 0, "sbert": 1, "combined": 2}[item["representation"]]):
        final_data.append([
            row["representation"].upper() if row["representation"] != "combined" else "Combined",
            f'{float(row["mae"]):.4f}', f'{float(row["mdae"]):.4f}',
            row["selected_epochs"], f'{int(row["input_dimensions"]):,}',
        ])
    story += [table(final_data, [4.7 * cm, 2.7 * cm, 2.7 * cm, 2.5 * cm, 3 * cm], alignments=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])]
    story += [Spacer(1, 0.2 * cm), Image(str(ROOT / "results/DM_final_metrics.png"), width=15.5 * cm, height=5.64 * cm), p("Figure 2. Final held-out MAE and MdAE; lower is better.", styles["caption"])]
    story += [p(
        "Combined embeddings achieved the lowest final error on both metrics. Against BERT, fusion reduced MAE by 0.3984 points (9.18%) and MdAE by 0.4203 (19.83%). Against SBERT, it reduced MAE by 0.4822 (10.90%) and MdAE by 0.3232 (15.98%).", styles["callout"])]

    story += [p("5. Discussion", styles["h1"])]
    story += [p(
        "The held-out result indicates complementary information: BERT captures contextual technical wording while SBERT emphasises whole-issue meaning. Concatenation made both views available to the regressor and improved future-issue predictions. However, validation evidence was nuanced because combined MdAE was not best. The final result supports fusion for this target period, not universal superiority.", styles["body"])]
    story += [p(
        "Combined final MAE (3.9416) exceeded MdAE (1.6988), indicating a subset of large errors. This is consistent with the right-skewed story-point distribution. Later folds generally had lower MAE, which may reflect larger training samples, temporal drift, or both.", styles["body"])]

    story += [p("6. Threats to Validity and Limitations", styles["h1"])]
    limitation_groups = [
        ("Internal validity", "One seed was evaluated; broad hyperparameter search was not performed; representation dimensions differed even though heads and partitions were shared."),
        ("Construct validity", "Story points are team-specific relative estimates; change-flag accuracy depends on TAWOS; descriptions beyond 256 tokens were truncated."),
        ("External validity", "The result covers DM only; encoders were frozen; open-source Jira history may not represent industrial projects."),
        ("Conclusion validity", "No repeated-seed confidence intervals or significance tests were conducted; one final temporal interval cannot describe all future periods."),
    ]
    for title, body in limitation_groups:
        story += [p(title, styles["h2"]), p(body, styles["body"])]

    story += [p("7. Reproducibility", styles["h1"])]
    story += [p(
        "The repository includes clean and executed notebooks, exact configuration, split metadata, 15 fold records, summaries, plots, and final metrics. The 4.31 GB source dump and large embedding/model caches are excluded. Reproduction requires TAWOS.sql at MyDrive/TAWOS_NLP/data/raw/, a GPU-enabled Colab runtime, and running all cells in order. DEMO.md provides the presentation and verification sequence.", styles["body"])]

    story += [p("8. Conclusion", styles["h1"])]
    story += [p(
        "On 2,957 later DM issues, combined embeddings achieved MAE 3.9416 and MdAE 1.6988, outperforming both individual representations. This supports complementary contextual and sentence-level features for the selected project. Future work should repeat the experiment across projects and seeds, test software-domain encoders, compare fusion architectures, and evaluate statistical reliability.", styles["body"])]

    story += [p("References", styles["h1"])]
    refs = [
        "1. Tawosi, V., Al-Subaihin, A., Moussa, R., and Sarro, F. A Versatile Dataset of Agile Open Source Software Projects. MSR 2022, 707–711. DOI: 10.1145/3524842.3528029.",
        "2. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.",
        "3. Reimers, N. and Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.",
        "4. TAWOS V1.1 dataset. DOI: 10.5522/04/21308124.",
    ]
    for ref in refs:
        story.append(p(ref, styles["small"]))

    story += [PageBreak(), p("Appendix A — Fold-level Validation Metrics", styles["h1"])]
    fold_data = [["Representation", "Fold", "MAE", "MdAE", "Best epoch"]]
    for row in folds:
        fold_data.append([
            row["representation"].upper() if row["representation"] != "combined" else "Combined",
            row["fold"], f'{float(row["mae"]):.4f}', f'{float(row["mdae"]):.4f}', row["best_epoch"],
        ])
    story += [table(fold_data, [4.5 * cm, 2.2 * cm, 3 * cm, 3 * cm, 2.8 * cm], alignments=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"], font_size=8.1)]
    story += [p("Appendix B — Recorded Configuration", styles["h1"])]
    config_data = [
        ["Setting", "Recorded value"],
        ["Project", experiment["project_key"]],
        ["Seed", str(experiment["seed"])],
        ["Device", experiment["device"]],
        ["BERT model", experiment["bert_model"]],
        ["SBERT model", experiment["sbert_model"]],
        ["Maximum tokens", str(experiment["max_length"])],
        ["Fusion", "Concatenation (768 + 384 = 1,152)"],
        ["Training", "AdamW; lr 0.001; wd 0.0001; batch 64"],
        ["Selection", "5 temporal folds; median best epoch"],
        ["Metrics", "MAE and MdAE"],
    ]
    story += [table(config_data, [4.8 * cm, 10.7 * cm], alignments=["LEFT", "LEFT"])]

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()

