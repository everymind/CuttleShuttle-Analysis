using Bonsai.Expressions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reactive.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace Bonsai.VideoAnnotations
{
    /// <summary>
    /// Represents a combinator that sorts the elements of the input enumerable
    /// sequences according to the specified key.
    /// </summary>
    [DefaultProperty("KeySelector")]
    [Description("Sorts the elements of the input enumerable sequences according to the specified key.")]
    public class SortByBuilder : SingleArgumentExpressionBuilder
    {
        /// <summary>
        /// Gets or sets a string used to specify a key for each element of the observable sequence.
        /// </summary>
        [Description("The inner properties that will be used as a key for sorting the elements of the sequence.")]
        [Editor("Bonsai.Design.MultiMemberSelectorEditor, Bonsai.Design", "System.Drawing.Design.UITypeEditor, System.Drawing, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a")]
        public string KeySelector { get; set; }

        internal static Expression MemberSelector(Expression expression, string selector)
        {
            var selectedMembers = SelectMembers(expression, selector).ToArray();
            if (selectedMembers.Length > 1)
            {
                return CreateTuple(selectedMembers);
            }
            else return selectedMembers.Single();
        }

        internal static Expression CreateTuple(Expression[] arguments)
        {
            return CreateTuple(arguments, 0);
        }

        internal static Expression CreateTuple(Expression[] arguments, int offset)
        {
            const int MaxLength = 7;
            var length = arguments.Length - offset;
            if (length > MaxLength)
            {
                var rest = CreateTuple(arguments, offset + MaxLength);
                var selectedArguments = new Expression[MaxLength + 1];
                selectedArguments[MaxLength] = rest;
                Array.Copy(arguments, offset, selectedArguments, 0, MaxLength);
                var memberTypes = Array.ConvertAll(selectedArguments, member => member.Type);
                var constructor = typeof(Tuple<,,,,,,,>).MakeGenericType(memberTypes).GetConstructors()[0];
                return Expression.New(constructor, selectedArguments);
            }
            else
            {
                if (offset > 0)
                {
                    var selectedArguments = new Expression[length];
                    Array.Copy(arguments, offset, selectedArguments, 0, length);
                    arguments = selectedArguments;
                }
                var memberTypes = Array.ConvertAll(arguments, member => member.Type);
                return Expression.Call(typeof(Tuple), "Create", memberTypes, arguments);
            }
        }

        /// <summary>
        /// Generates an <see cref="Expression"/> node from a collection of input arguments.
        /// The result can be chained with other builders in a workflow.
        /// </summary>
        /// <param name="arguments">
        /// A collection of <see cref="Expression"/> nodes that represents the input arguments.
        /// </param>
        /// <returns>An <see cref="Expression"/> tree node.</returns>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            var source = arguments.First();
            var parameterType = source.Type.GetGenericArguments()[0];
            var enumerableType = parameterType.GetGenericArguments()[0];

            var parameter = Expression.Parameter(enumerableType);
            var keySelectorBody = MemberSelector(parameter, KeySelector);
            var keySelectorLambda = Expression.Lambda(keySelectorBody, parameter);
            var combinator = Expression.Constant(this);
            return Expression.Call(
                combinator,
                "Process",
                new[] { parameter.Type, keySelectorLambda.ReturnType },
                source,
                keySelectorLambda);
        }

        IObservable<IEnumerable<TSource>> Process<TSource, TKey>(IObservable<IEnumerable<TSource>> source, Func<TSource, TKey> keySelector)
        {
            return source.Select(input => input.OrderBy(keySelector));
        }
    }
}
