import * as react_jsx_runtime from 'react/jsx-runtime';
import { WindowProps } from './props.js';
import '@copilotkit/runtime-client-gql';
import '../../types/suggestions.js';
import 'react';

declare const Window: ({ children, clickOutsideToClose, shortcut, hitEscapeToClose, }: WindowProps) => react_jsx_runtime.JSX.Element;

export { Window };
