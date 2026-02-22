import * as react_jsx_runtime from 'react/jsx-runtime';
import { RenderSuggestionsListProps } from './props.js';
import '@copilotkit/runtime-client-gql';
import '../../types/suggestions.js';
import 'react';

declare function Suggestions({ suggestions, onSuggestionClick }: RenderSuggestionsListProps): react_jsx_runtime.JSX.Element;

export { Suggestions };
