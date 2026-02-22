import * as react_jsx_runtime from 'react/jsx-runtime';
import { CopilotModalProps } from './Modal.js';
import 'react';
import './props.js';
import '@copilotkit/runtime-client-gql';
import '../../types/suggestions.js';
import './Chat.js';
import './ChatContext.js';
import '@copilotkit/react-core';

declare function CopilotPopup(props: CopilotModalProps): react_jsx_runtime.JSX.Element;

export { CopilotPopup };
