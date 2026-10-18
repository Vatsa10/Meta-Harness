"""Authored research-desk requests over the Constitution corpus.

Each request has a short gold answer (`accept` lists the spellings the grader takes), the source
document, and the gold ticket (or None when no ticket is due). The ticket rules the agent is given
(settings.TICKET_POLICY) decide the gold fields deterministically:
  queue: eligibility (a named person fails an office qualification on the stated date), rights (a
  reported violation of a Bill of Rights guarantee), compliance (a government body plans something a
  provision forbids); doc_id: the governing provision; priority: "high" when the stated deadline is
  within 14 days of the request date, else "normal".

Requests need more than a look-up: ages and years are computed from dates, some answers combine two
provisions, and several mention a clause that does NOT apply (a distractor).
"""


def T(text, answer, accept, source, ticket=None):
    return {"input": text, "answer": answer, "accept": accept, "source": source, "ticket": ticket}


def tk(queue, doc_id, priority="normal"):
    return {"queue": queue, "doc_id": doc_id, "priority": priority}


def n(x, *words):
    return [str(x), *words]


TASKS = [
    # --- eligibility with date arithmetic (answer: full years short of the requirement; 0 if eligible)
    T("Request dated 2026-10-01. Ana Ruiz, born 2002-03-14, would be sworn into the House on 2027-01-03. "
      "How many full years short of the minimum age for a Representative will she be on that day? Answer 0 if she qualifies.",
      "1", n(1, "one"), "art1-sec2", tk("eligibility", "art1-sec2")),
    T("Request dated 2026-10-01. Mark Chu, born 1995-12-30, would take a Senate seat on 2027-01-03. "
      "How many full years short of the Senate minimum age will he be? Answer 0 if he qualifies.",
      "0", n(0, "zero"), "art1-sec3"),
    T("Request dated 2026-10-05. Lena Ortiz became a citizen on 2019-06-01 and would take a Senate seat on 2027-01-03. "
      "How many full years of citizenship will she have on that day?",
      "7", n(7, "seven"), "art1-sec3", tk("eligibility", "art1-sec3")),
    T("Request dated 2026-10-05. Omar Haddad became a citizen on 2016-02-10 and would join the House on 2027-01-03. "
      "How many full years of citizenship will he have on that day?",
      "10", n(10, "ten"), "art1-sec2"),
    T("Request dated 2026-10-10. Governor Ray Holt, born 1993-02-02, would be inaugurated President on 2029-01-20. "
      "How old will he be on inauguration day?",
      "35", n(35, "thirty five"), "art2-sec1"),
    T("Request dated 2026-10-10. Priya Nair, born 1994-08-15, would be inaugurated President on 2029-01-20. "
      "How old will she be on inauguration day?",
      "34", n(34, "thirty four"), "art2-sec1", tk("eligibility", "art2-sec1")),
    T("Request dated 2026-10-12. Dana Wu, born 1997-10-20, must file Senate papers by 2026-10-20 to be seated 2027-01-03. "
      "How many full years short of the Senate minimum age will she be when seated? Answer 0 if she qualifies.",
      "1", n(1, "one"), "art1-sec3", tk("eligibility", "art1-sec3", "high")),
    T("Request dated 2026-10-12. Sam Reed became a citizen on 2021-03-01; the House filing deadline is 2026-10-19 and he would be seated 2027-01-03. "
      "How many more full years of citizenship would he need on the seating day to qualify?",
      "2", n(2, "two"), "art1-sec2", tk("eligibility", "art1-sec2", "high")),
    T("Request dated 2026-10-14. Eva Lind, born 1990-01-01, has lived in the United States since 2015-01-01 and would be inaugurated President on 2029-01-20. "
      "How many full years of U.S. residence will she have on that day?",
      "14", n(14, "fourteen"), "art2-sec1"),
    T("Request dated 2026-10-14. Tom Berg, born 1980-05-05, has lived in the United States since 2016-07-01 and would be inaugurated President on 2029-01-20. "
      "How many more full years of residence would he need to qualify?",
      "2", n(2, "two"), "art2-sec1", tk("eligibility", "art2-sec1")),
    # --- multi-hop reference numbers
    T("How many more years of citizenship does a Senator need than a Representative?", "2", n(2, "two"), "art1-sec3"),
    T("What is the sum of the minimum ages for Representative, Senator and President?", "90", n(90, "ninety"), "art2-sec1"),
    T("A Senator serves a full term. How many two-year House terms fit into one Senate term?", "3", n(3, "three"), "art1-sec3"),
    T("What fraction of Senators present must concur for the President to make a treaty? Answer as a fraction.",
      "two thirds", ["two thirds", "2/3", "twothirds"], "art2-sec2"),
    T("If a bill passes the House with 290 of 435 members voting yes on reconsideration after a veto, does it meet the override threshold of that House? "
      "Answer yes or no.", "yes", ["yes"], "art1-sec7"),
    T("A bill reaches the President, who objects to it. Excluding Sundays, how many days does he have to return it before it becomes law anyway "
      "(Congress in session)? Then add the number of Senators per State. Give the sum.",
      "12", n(12, "twelve"), "art1-sec7"),
    T("How many witnesses to the same overt act are required for a treason conviction when there is no confession in open court?",
      "2", n(2, "two"), "art3-sec3"),
    T("With 50 States, how many States must ratify an amendment for it to take effect? Give the number.",
      "38", n(38, "thirty eight"), "art5"),
    T("How many of the original thirteen States had to ratify for the Constitution to be established between them?",
      "9", n(9, "nine"), "art7"),
    T("If the first enumeration was in 1790 and later ones follow the minimum interval, in which year was the third enumeration?",
      "1810", n(1810), "art1-sec2"),
    T("Under the original ratio, what was the maximum number of Representatives for a State of 1,200,000 persons?",
      "40", n(40, "forty"), "art1-sec2"),
    T("During a session, the House wants to adjourn from Monday 2026-04-06 to Friday 2026-04-10 without the Senate's consent. "
      "How many days is the longest adjournment allowed without consent?",
      "3", n(3, "three"), "art1-sec5"),
    T("A civil suit at common law claims $15. Is a jury trial guaranteed by the Bill of Rights? Answer yes or no.",
      "no", ["no"], "amend7"),
    T("Into how many classes were Senators divided after the first election, and so what fraction of seats is chosen every second year? Answer the fraction.",
      "one third", ["one third", "1/3", "onethird"], "art1-sec3"),
    # --- rights complaints
    T("Request dated 2026-10-02. A homeowner reports soldiers were quartered in his house in peacetime without his consent; "
      "his hearing is on 2026-11-30. Which amendment applies? Give the number.",
      "3", n(3, "three", "third"), "amend3", tk("rights", "amend3")),
    T("Request dated 2026-10-02. A tenant reports police searched her apartment and seized papers with no warrant; "
      "her lawyer needs a memo by 2026-10-09. Which amendment applies? Give the number.",
      "4", n(4, "four", "fourth"), "amend4", tk("rights", "amend4", "high")),
    T("Request dated 2026-10-03. A defendant says he was never informed of the nature of the accusation against him and was refused counsel. "
      "Which amendment applies? Give the number.",
      "6", n(6, "six", "sixth"), "amend6", tk("rights", "amend6")),
    T("Request dated 2026-10-03. A detainee's bail was set at $5 million for a minor offense; the review is on 2026-10-06. "
      "Which amendment applies? Give the number.",
      "8", n(8, "eight", "eighth"), "amend8", tk("rights", "amend8", "high")),
    T("A student asks which amendment protects against being tried twice for the same offense. Give the number.",
      "5", n(5, "five", "fifth"), "amend5"),
    T("A journalist asks which amendment protects freedom of the press, for an article (no complaint). Give the number.",
      "1", n(1, "one", "first"), "amend1"),
    # --- compliance (a government body plans something forbidden; distractors included)
    T("Request dated 2026-10-04. The state of Delaware plans to coin its own silver money starting 2027-01-01. "
      "Which section of Article 1 forbids this? Give the section number.",
      "10", n(10, "ten"), "art1-sec10", tk("compliance", "art1-sec10")),
    T("Request dated 2026-10-04. A federal agency proposes granting a title of nobility to a donor at a ceremony on 2026-10-15. "
      "Which section of Article 1 forbids this? Give the section number.",
      "9", n(9, "nine"), "art1-sec9", tk("compliance", "art1-sec9", "high")),
    T("Request dated 2026-10-06. A state legislature votes on 2026-10-13 on a tax on goods exported from that state. "
      "Which section of Article 1 bars taxes or duties on articles exported from any State? Give the section number.",
      "9", n(9, "nine"), "art1-sec9", tk("compliance", "art1-sec9", "high")),
    T("Request dated 2026-10-06. A state wants to sign a treaty with a foreign nation next spring. "
      "Which section of Article 1 forbids this? Give the section number.",
      "10", n(10, "ten"), "art1-sec10", tk("compliance", "art1-sec10")),
    T("Request dated 2026-10-07. Congress plans to regulate commerce with foreign nations under a new act. "
      "Which section of Article 1 grants this power? Give the section number.",
      "8", n(8, "eight"), "art1-sec8"),
    T("Request dated 2026-10-07. A state plans to keep troops in time of peace without the consent of Congress, starting 2026-10-20. "
      "Which section of Article 1 forbids this? Give the section number.",
      "10", n(10, "ten"), "art1-sec10", tk("compliance", "art1-sec10", "high")),
]
