import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const workbookPath = path.join(
  process.cwd(),
  "outputs",
  "data_splits",
  "data_splits_review.xlsx",
);
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const expectedSheets = [
  "Overview",
  "Well Assignment",
  "Split Summary",
  "Concentration Balance",
  "Patch Outer Fold",
  "Nested Roles",
  "Protocol",
];
const sheetNames = Array.from(workbook.worksheets).map((sheet) => sheet.name);
const errors = [];

for (const sheetName of expectedSheets) {
  if (!sheetNames.includes(sheetName)) errors.push(`Missing sheet: ${sheetName}`);
}
if (sheetNames[0] !== "Overview") errors.push("Overview is not the first sheet");

const overview = workbook.worksheets.getItem("Overview");
const overviewValues = overview.getRange("B4:B9").values.flat().map(Number);
const expectedOverview = [1824, 1056, 768, 96, 5, 240920];
for (let index = 0; index < expectedOverview.length; index += 1) {
  if (overviewValues[index] !== expectedOverview[index]) {
    errors.push(`Unexpected Overview value at B${index + 4}: ${overviewValues[index]}`);
  }
}
const foldValues = overview.getRange("A13:F17").values.map((row) => row.map(Number));
const expectedFolds = [
  [1, 60, 16, 20, 8, 12],
  [2, 61, 16, 19, 7, 12],
  [3, 61, 16, 19, 7, 12],
  [4, 61, 16, 19, 7, 12],
  [5, 61, 16, 19, 7, 12],
];
for (let row = 0; row < expectedFolds.length; row += 1) {
  for (let column = 0; column < expectedFolds[row].length; column += 1) {
    if (foldValues[row][column] !== expectedFolds[row][column]) {
      errors.push(
        `Unexpected fold summary at row ${row + 13}, column ${column + 1}: ${foldValues[row][column]}`,
      );
    }
  }
}

const wellSheet = workbook.worksheets.getItem("Well Assignment");
const wellRows = wellSheet.getRange("A2:J97").values;
if (wellRows.length !== 96) errors.push(`Expected 96 well rows, found ${wellRows.length}`);
if (new Set(wellRows.map((row) => row[0])).size !== 96) {
  errors.push("Well Assignment contains duplicate well_id values");
}
const outerFoldCounts = new Map();
for (const row of wellRows) {
  const fold = Number(row[4]);
  outerFoldCounts.set(fold, (outerFoldCounts.get(fold) ?? 0) + 1);
}
const expectedOuterCounts = new Map([[1, 20], [2, 19], [3, 19], [4, 19], [5, 19]]);
for (const [fold, count] of expectedOuterCounts) {
  if (outerFoldCounts.get(fold) !== count) {
    errors.push(`Unexpected outer fold ${fold} well count: ${outerFoldCounts.get(fold)}`);
  }
}

const splitRows = workbook.worksheets.getItem("Split Summary").getRange("A2:J51").values;
if (splitRows.length !== 50) errors.push(`Expected 50 split summary rows, found ${splitRows.length}`);
const balanceRows = workbook.worksheets
  .getItem("Concentration Balance")
  .getRange("A2:I96").values;
if (balanceRows.length !== 95) {
  errors.push(`Expected 95 concentration balance rows, found ${balanceRows.length}`);
}
for (const row of balanceRows) {
  if (Number(row[4]) + Number(row[5]) !== 96) {
    errors.push(`ML concentration balance error in fold ${row[0]} ${row[1]} level ${row[2]}`);
    break;
  }
  if (Number(row[6]) + Number(row[7]) + Number(row[8]) !== 96) {
    errors.push(`DL concentration balance error in fold ${row[0]} ${row[1]} level ${row[2]}`);
    break;
  }
}

const patchRows = workbook.worksheets
  .getItem("Patch Outer Fold")
  .getRange("A2:A1825").values.flat();
if (patchRows.filter((value) => value !== null).length !== 1824) {
  errors.push("Patch Outer Fold does not contain 1,824 patch_id values");
}
const nestedRows = workbook.worksheets
  .getItem("Nested Roles")
  .getRange("A2:A9121").values.flat();
if (nestedRows.filter((value) => value !== null).length !== 9120) {
  errors.push("Nested Roles does not contain 9,120 assignment rows");
}

const formulaErrorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "workbook formula error scan",
});
if (!formulaErrorScan.ndjson.includes("matched 0 entries")) {
  errors.push(`Formula error scan returned matches: ${formulaErrorScan.ndjson}`);
}

const report = {
  workbook: workbookPath,
  sheets: sheetNames,
  overview_values: overviewValues,
  fold_values: foldValues,
  well_rows: wellRows.length,
  split_summary_rows: splitRows.length,
  concentration_balance_rows: balanceRows.length,
  patch_rows: patchRows.filter((value) => value !== null).length,
  nested_rows: nestedRows.filter((value) => value !== null).length,
  errors,
};
console.log(JSON.stringify(report, null, 2));
if (errors.length) process.exitCode = 1;
