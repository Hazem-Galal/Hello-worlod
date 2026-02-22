import * as react_jsx_runtime from 'react/jsx-runtime';
import { CopilotContextParams, CopilotMessagesContextParams, CopilotChatSuggestionConfiguration } from '@copilotkit/react-core';

interface SuggestionsProps {
    title: string;
    message: string;
    partial?: boolean;
    className?: string;
    onClick: () => void;
}
declare function Suggestion({ title, onClick, partial, className }: SuggestionsProps): react_jsx_runtime.JSX.Element;
declare const reloadSuggestions: (context: CopilotContextParams & CopilotMessagesContextParams, chatSuggestionConfiguration: {
    [key: string]: CopilotChatSuggestionConfiguration;
}, setCurrentSuggestions: (suggestions: {
    title: string;
    message: string;
}[]) => void, abortControllerRef: React.MutableRefObject<AbortController | null>) => Promise<void>;

export { Suggestion, reloadSuggestions };
