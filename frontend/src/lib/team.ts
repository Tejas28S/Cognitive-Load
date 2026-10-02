/**
 * ============================================================================
 * TEAM DATA
 * ============================================================================
 * Names, roles and focus areas confirmed by the team. Institutional
 * affiliation is deliberately not shown; add an `affiliation` field back
 * here (and in Bios.tsx's MemberCard) if that's ever needed.
 *
 * `placeholder: true` marks an entry as unconfirmed and drives the amber
 * "unconfirmed" treatment in the UI. Leave it `false` once a person's real
 * details are in; flip a new/incomplete entry back to `true` rather than
 * inventing details for it.
 */

export type TeamMember = {
  name: string;
  role: string;
  /** One or two sentences. What they actually worked on. */
  focus: string;
  /** Optional: research interests, prior work, links. */
  interests?: string;
  /** True while any field is still unconfirmed. Drives the UI warning. */
  placeholder: boolean;
};

export const TEAM: TeamMember[] = [
  {
    name: "Tejas S",
    role: "Model Development",
    focus:
      "Worked on the EEG/fNIRS fusion model and the leakage-audited LOSO evaluation protocol.",
    placeholder: false,
  },
  {
    name: "Tejas MS",
    role: "Backend & Serving API",
    focus:
      "Worked on the FastAPI serving layer that exposes the trained model and powers the live demo.",
    placeholder: false,
  },
  {
    name: "Tanisha Mangal",
    role: "Frontend & UI Design",
    focus:
      "Worked on the React/TypeScript frontend, the site's page structure and the visual design system used across the project.",
    placeholder: false,
  },
  {
    name: "Skanda Prabhakar",
    role: "Data Pipeline & Preprocessing",
    focus:
      "Worked on data preprocessing and feature extraction from the raw EEG/fNIRS recordings.",
    placeholder: false,
  },
];

/** Supervisor / advisor entries, if the project has them. */
export const ADVISORS: TeamMember[] = [
  {
    name: "Prof. Chandravva Hebbi",
    role: "Project Supervisor",
    focus:
      "Provided academic supervision and guidance throughout the project, reviewing the methodology, evaluation protocol, and project direction.",
    placeholder: false,
  },
];

/** True while anything on the page still needs confirming. */
export const HAS_PLACEHOLDERS = [...TEAM, ...ADVISORS].some(
  (m) => m.placeholder
);
