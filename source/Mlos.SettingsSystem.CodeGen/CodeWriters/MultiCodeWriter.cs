// -----------------------------------------------------------------------
// <copyright file="MultiCodeWriter.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root
// for license information.
// </copyright>
// -----------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Mlos.SettingsSystem.CodeGen.CodeWriters
{
    /// <summary>
    /// Class for dispatching source types discovered through reflection to
    /// each of the registered code writers.
    /// </summary>
    internal class MultiCodeWriter : CodeWriter
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MultiCodeWriter"/> class.
        /// </summary>
        /// <param name="writers"></param>
        public MultiCodeWriter(IEnumerable<CodeWriter> writers)
        {
            codeWriters = writers.ToList();

            // Set current namespace to empty for all codewriters.
            //
            codeWriters.ForEach(r => codeWriterCurrentNamespace[r] = string.Empty);
        }

        /// <inheritdoc />
        public override bool Accept(Type sourceType)
        {
            acceptedCodeWriters = codeWriters.Where(r => r.Accept(sourceType)).ToList();

            return true;
        }

        /// <inheritdoc />
        public override void WriteBeginFile()
        {
            codeWriters.ForEach(r => r.WriteBeginFile());
        }

        /// <inheritdoc />
        public override void WriteEndFile()
        {
            // Close open type namespace.
            //
            codeWriters.ForEach(r =>
            {
                string currentNamespace = codeWriterCurrentNamespace[r];

                if (!string.IsNullOrEmpty(currentNamespace))
                {
                    r.WriteCloseTypeNamespace(currentNamespace);
                }

                codeWriterCurrentNamespace[r] = string.Empty;
            });

            codeWriters.ForEach(r => r.WriteEndFile());
        }

        /// <inheritdoc />
        public override void WriteOpenTypeNamespace(string @namespace)
        {
            // If required set correct namespace.
            //
            acceptedCodeWriters.ForEach(r =>
            {
                string currentNamespace = codeWriterCurrentNamespace[r];
                if (currentNamespace != @namespace)
                {
                    if (!string.IsNullOrEmpty(currentNamespace))
                    {
                        r.WriteCloseTypeNamespace(currentNamespace);
                    }

                    r.WriteOpenTypeNamespace(@namespace);
                    codeWriterCurrentNamespace[r] = @namespace;
                }
            });
        }

        /// <inheritdoc />
        public override void WriteCloseTypeNamespace(string @namespace)
        {
            // Do not close the namespace. Current namespace will be closed on new type definition or file close.
            //
        }

        #region Per class methods

        /// <inheritdoc />
        public override void BeginVisitType(Type sourceType)
        {
            acceptedCodeWriters.ForEach(r => r.BeginVisitType(sourceType));
        }

        /// <inheritdoc />
        public override void WriteComments(CodeComment codeComment)
        {
            acceptedCodeWriters.ForEach(r => r.WriteComments(codeComment));
        }

        /// <inheritdoc />
        public override void EndVisitType(Type sourceType)
        {
            acceptedCodeWriters.ForEach(r => r.EndVisitType(sourceType));
        }

        /// <inheritdoc />
        public override void VisitField(CppField cppField)
        {
            acceptedCodeWriters.ForEach(r => r.VisitField(cppField));
        }

        /// <inheritdoc />
        public override void VisitConstField(CppConstField cppConstField)
        {
            acceptedCodeWriters.ForEach(r => r.VisitConstField(cppConstField));
        }

        #endregion

        /// <summary>
        /// Gets the code generated by each of the registered code writers.
        /// </summary>
        /// <returns></returns>
        public Dictionary<string, StringBuilder> GetOutput()
        {
            // Dictionary containing generated source code.
            // Key is file postfix, value is the code.
            //
            var result = new Dictionary<string, StringBuilder>();

            foreach (CodeWriter codeWriter in codeWriters)
            {
                string postfix = codeWriter.FilePostfix;
                if (!result.ContainsKey(postfix))
                {
                    result[postfix] = new StringBuilder();
                }

                result[postfix].Append(codeWriter.GetGeneratedString());
            }

            return result;
        }

        private readonly List<CodeWriter> codeWriters;

        /// <summary>
        /// Keeps current namespace for file. Used to reduce number of opening and closing namespace statements.
        /// </summary>
        private readonly Dictionary<CodeWriter, string> codeWriterCurrentNamespace = new Dictionary<CodeWriter, string>();

        private List<CodeWriter> acceptedCodeWriters;

        /// <inheritdoc />
        public override string FilePostfix => throw new NotImplementedException();
    }
}
