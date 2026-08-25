/**
 * ============================================================================
 * TEAM DATA — CONTAINS PLACEHOLDERS. REPLACE BEFORE PUBLISHING.
 * ============================================================================
 *
 * Everything marked PLACEHOLDER below was NOT invented from research material
 * and must be confirmed by a person before this page goes anywhere public.
 * Real names, roles and affiliations are the one category of content on this
 * site that cannot be derived from the repository.
 *
 * To finish this page:
 *   1. Fill in every field whose value contains "[PLACEHOLDER".
 *   2. Delete the `placeholder: true` flag from each completed entry — that
 *      flag is what renders the amber "unconfirmed" treatment in the UI.
 *   3. Delete any surplus template entries you don't need.
 *
 * The single entry with a real name comes from this repository's git author
 * record, not from guesswork; its role and affiliation are still unconfirmed.
 */

export type TeamMember = {
  name: string;
  role: string;
  affiliation: string;
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
    role: "[PLACEHOLDER, confirm role]",
    affiliation: "[PLACEHOLDER, confirm institution / department]",
    focus:
      "[PLACEHOLDER, confirm contribution. Git history attributes the model, serving API and research framework to this author, but the split of work across the team has not been confirmed.]",
    placeholder: true,
  },
  {
    name: "[PLACEHOLDER, team member 2]",
    role: "[PLACEHOLDER, confirm role]",
    affiliation: "[PLACEHOLDER, confirm institution / department]",
    focus: "[PLACEHOLDER, confirm contribution]",
    placeholder: true,
  },
  {
    name: "[PLACEHOLDER, team member 3]",
    role: "[PLACEHOLDER, confirm role]",
    affiliation: "[PLACEHOLDER, confirm institution / department]",
    focus: "[PLACEHOLDER, confirm contribution]",
    placeholder: true,
  },
];

/** Supervisor / advisor entries, if the project has them. */
export const ADVISORS: TeamMember[] = [
  {
    name: "[PLACEHOLDER, supervisor name]",
    role: "[PLACEHOLDER, e.g. Project Supervisor]",
    affiliation: "[PLACEHOLDER, confirm institution / department]",
    focus: "[PLACEHOLDER, confirm advisory role]",
    placeholder: true,
  },
];

/** True while anything on the page still needs confirming. */
export const HAS_PLACEHOLDERS = [...TEAM, ...ADVISORS].some(
  (m) => m.placeholder
);
