import { readdirSync, statSync, writeFileSync } from "node:fs";
import type { Stats } from "node:fs";
import path from "node:path";
import ts from "typescript";

export interface EnrichOptions {
  readonly dryRun?: boolean;
  readonly logger?: (message: string) => void;
}

const SUPPORTED_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx"]);

export function enrichJsDoc(
  entry: string,
  options: EnrichOptions = {},
): string[] {
  const resolved = path.resolve(entry);
  const stats = statSync(resolved, { throwIfNoEntry: false });
  if (!stats) {
    throw new Error(`Path not found: ${resolved}`);
  }

  const targets = collectTargets(resolved, stats);
  const updated: string[] = [];
  for (const filePath of targets) {
    const changed = processFile(filePath, options);
    if (changed) {
      updated.push(filePath);
    }
  }

  if (options.logger) {
    if (updated.length === 0) {
      options.logger(
        `No exported functions required documentation updates under ${resolved}`,
      );
    } else {
      options.logger(`Added documentation to ${updated.length} file(s).`);
      updated.forEach((file) => options.logger?.(`  • ${file}`));
    }
  }

  return updated;
}

function collectTargets(resolved: string, stats: Stats): string[] {
  if (stats.isDirectory()) {
    const results: string[] = [];
    for (const entry of readdirSync(resolved)) {
      if (entry.startsWith(".")) {
        continue;
      }
      const child = path.join(resolved, entry);
      const childStats = statSync(child, { throwIfNoEntry: false }) as
        | Stats
        | undefined;
      if (!childStats) {
        continue;
      }
      if (childStats.isDirectory()) {
        results.push(...collectTargets(child, childStats));
      } else if (SUPPORTED_EXTENSIONS.has(path.extname(child))) {
        results.push(child);
      }
    }
    return results;
  }

  if (SUPPORTED_EXTENSIONS.has(path.extname(resolved))) {
    return [resolved];
  }

  return [];
}

function processFile(filePath: string, options: EnrichOptions): boolean {
  const configuration: ts.CompilerOptions = {
    allowJs: true,
    target: ts.ScriptTarget.ES2020,
    module: ts.ModuleKind.CommonJS,
  };
  const program = ts.createProgram([filePath], configuration);
  const source = program.getSourceFile(filePath);
  if (!source) {
    throw new Error(`Unable to read ${filePath}`);
  }
  const checker = program.getTypeChecker();
  const printer = ts.createPrinter({ newLine: ts.NewLineKind.LineFeed });
  let mutated = false;

  const statements: ts.Statement[] = source.statements.map((statement) => {
    if (
      ts.isFunctionDeclaration(statement) &&
      statement.body &&
      isNodeExported(statement)
    ) {
      if (hasJsDoc(statement)) {
        return statement;
      }
      mutated = true;
      return addJsDocToFunction(statement, source, checker);
    }

    if (ts.isVariableStatement(statement) && isNodeExported(statement)) {
      const updated = maybeAnnotateVariableStatement(
        statement,
        source,
        checker,
      );
      if (updated !== statement) {
        mutated = true;
        return updated;
      }
    }

    return statement;
  });

  if (!mutated) {
    return false;
  }

  const transformed = ts.factory.updateSourceFile(
    source,
    statements,
    source.isDeclarationFile,
    source.referencedFiles,
    source.typeReferenceDirectives,
    source.hasNoDefaultLib,
    source.libReferenceDirectives,
  );

  const output = printer.printFile(transformed);
  if (!options.dryRun) {
    writeFileSync(filePath, output, "utf8");
  }
  return true;
}

function hasJsDoc(node: ts.Node): boolean {
  return ts.getJSDocCommentsAndTags(node).length > 0;
}

function isNodeExported(
  node: ts.FunctionDeclaration | ts.VariableStatement,
): boolean {
  const flags = ts.getCombinedModifierFlags(node as ts.Declaration);
  if ((flags & ts.ModifierFlags.Export) !== 0) {
    return true;
  }

  if (ts.isFunctionDeclaration(node) && node.name) {
    const name = node.name.text;
    const source = node.getSourceFile();
    return source.statements.some((statement) => {
      if (ts.isExportDeclaration(statement) && statement.exportClause) {
        const clause = statement.exportClause;
        if (ts.isNamedExports(clause)) {
          return clause.elements.some((element) => element.name.text === name);
        }
      }
      return (
        ts.isExportAssignment(statement) &&
        ts.isIdentifier(statement.expression) &&
        statement.expression.text === name
      );
    });
  }

  if (ts.isVariableStatement(node)) {
    const identifiers = node.declarationList.declarations
      .map((declaration) =>
        ts.isIdentifier(declaration.name) ? declaration.name.text : undefined,
      )
      .filter((value): value is string => value !== undefined);
    if (identifiers.length > 0) {
      const source = node.getSourceFile();
      return source.statements.some(
        (statement) =>
          ts.isExportDeclaration(statement) &&
          statement.exportClause !== undefined &&
          ts.isNamedExports(statement.exportClause) &&
          statement.exportClause.elements.some((element) =>
            identifiers.includes(element.name.text),
          ),
      );
    }
  }

  return false;
}

function addJsDocToFunction(
  node: ts.FunctionDeclaration,
  source: ts.SourceFile,
  checker: ts.TypeChecker,
): ts.FunctionDeclaration {
  const doc = createJsDocComment(
    node.name?.getText(source) ?? "function",
    node.parameters,
    node.type,
    source,
    checker,
    node,
  );
  const updated = ts.factory.updateFunctionDeclaration(
    node,
    node.modifiers,
    node.asteriskToken,
    node.name,
    node.typeParameters,
    node.parameters,
    node.type,
    node.body,
  );
  return withJsDoc(updated, doc);
}

function maybeAnnotateVariableStatement(
  node: ts.VariableStatement,
  source: ts.SourceFile,
  checker: ts.TypeChecker,
): ts.VariableStatement {
  const declarations = node.declarationList.declarations;
  if (declarations.length !== 1) {
    return node;
  }
  const declaration = declarations[0];
  const initializer = declaration.initializer;
  if (!initializer) {
    return node;
  }

  if (ts.isFunctionExpression(initializer) || ts.isArrowFunction(initializer)) {
    if (hasJsDoc(node)) {
      return node;
    }
    const name = declaration.name.getText(source);
    const parameters = initializer.parameters;
    const returnType = ts.isFunctionExpression(initializer)
      ? initializer.type
      : (initializer.type ?? declaration.type ?? undefined);

    const doc = createJsDocComment(
      name,
      parameters,
      returnType,
      source,
      checker,
      initializer,
    );
    const updated = ts.factory.updateVariableStatement(
      node,
      node.modifiers,
      ts.factory.updateVariableDeclarationList(node.declarationList, [
        ts.factory.updateVariableDeclaration(
          declaration,
          declaration.name,
          declaration.exclamationToken,
          declaration.type,
          initializer,
        ),
      ]),
    );
    return withJsDoc(updated, doc);
  }

  return node;
}

function createJsDocComment(
  functionName: string,
  parameters: readonly ts.ParameterDeclaration[],
  returnType: ts.TypeNode | undefined,
  source: ts.SourceFile,
  checker: ts.TypeChecker,
  returnFallback: ts.Node | undefined,
): string {
  const header = `Auto-generated documentation for ${functionName}.`;
  const lines: string[] = [header];

  for (const parameter of parameters) {
    const name = parameter.name.getText(source);
    const type = describeType(parameter.type, parameter, checker);
    lines.push(
      `@param ${type ? `{${type}} ` : ""}${name} - Parameter ${name}.`,
    );
  }

  const returnDescription = describeType(returnType, returnFallback, checker);
  lines.push(
    `@returns ${returnDescription ? `{${returnDescription}} ` : ""}Return value.`,
  );

  return formatJsDocLines(lines);
}

function describeType(
  typeNode: ts.TypeNode | undefined,
  fallbackNode: ts.Node | undefined,
  checker: ts.TypeChecker,
): string | undefined {
  if (typeNode) {
    return typeNode.getText().trim();
  }
  if (fallbackNode) {
    try {
      const type = checker.getTypeAtLocation(fallbackNode);
      const text = checker.typeToString(type);
      return text === "any" ? undefined : text;
    } catch (error) {
      return undefined;
    }
  }
  return undefined;
}

function formatJsDocLines(lines: string[]): string {
  const body = lines
    .map((line) => (line.length === 0 ? " *" : ` * ${line}`))
    .join("\n");
  return `*\n${body}\n `;
}

function withJsDoc<T extends ts.Node>(node: T, doc: string): T {
  return ts.addSyntheticLeadingComment(
    node,
    ts.SyntaxKind.MultiLineCommentTrivia,
    doc,
    true,
  );
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error(
      "Usage: ts-node tasks/documentation/jsdoc_enricher.ts <path> [--dry-run]",
    );
    process.exitCode = 1;
    process.exit(1);
  }
  const dryRun = args.includes("--dry-run");
  const targets = args.filter((arg) => !arg.startsWith("--"));
  for (const target of targets) {
    enrichJsDoc(target, {
      dryRun,
      logger: (message) => console.log(message),
    });
  }
}
