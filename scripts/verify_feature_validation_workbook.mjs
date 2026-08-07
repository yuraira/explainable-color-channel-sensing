import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const workbookPath = path.join(
  process.cwd(),
  "outputs",
  "feature_validation",
  "feature_validation_review.xlsx",
);
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const expectedSheets = [
  "Overview",
  "Concentration Summary",
  "Trends",
  "Adjacent Overlap",
  "Variability",
  "Position Bias",
  "Position Well",
  "Background Control",
  "Feature Correlation",
  "Model Feature Sets",
];
const sheetNames = Array.from(workbook.worksheets).map((sheet) => sheet.name);
const errors = [];
for (const sheetName of expectedSheets) {
  if (!sheetNames.includes(sheetName)) errors.push(`Missing sheet: ${sheetName}`);
}

const overview = workbook.worksheets.getItem("Overview");
const overviewValues = overview.getRange("B4:B10").values.flat();
const expectedOverview = [19, 11, 8, 3, 4, 2.1805645081128757, 0];
for (let index = 0; index < expectedOverview.length; index += 1) {
  if (Math.abs(Number(overviewValues[index]) - expectedOverview[index]) > 1e-6) {
    errors.push(
      `Unexpected Overview value at B${index + 4}: ${overviewValues[index]}`,
    );
  }
}
if (overviewValues.some((value) => typeof value === "string" && value.startsWith("#"))) {
  errors.push("Formula error displayed on Overview sheet");
}

const trends = workbook.worksheets.getItem("Trends");
const trendRows = trends.getRange("A2:J31").values;
if (trendRows.length !== 30) errors.push(`Expected 30 trend rows, found ${trendRows.length}`);
if (!trendRows.every((row) => typeof row[4] === "number")) {
  errors.push("Trend rho values are not numeric");
}

const report = {
  workbook: workbookPath,
  sheets: sheetNames,
  overview_values: overviewValues,
  trend_rows: trendRows.length,
  errors,
};
console.log(JSON.stringify(report, null, 2));
if (errors.length) process.exitCode = 1;
