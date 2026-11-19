import table from "../shared/exercises.json";

/** The exercise table. The master is backend/app/pipeline/rep_counter.EXERCISES; scripts/export_exercises.py writes it to
 *  src/shared/exercises.json (a Python test fails when the two differ), so the picker and the engine see the same numbers. */
export type ExerciseCfg = { joints: Record<string, string>; rest: number; peak: number };
export const EXERCISES = table as Record<string, ExerciseCfg>;
export const EXERCISE_NAMES = Object.keys(EXERCISES);        // JSON keeps the table's order
