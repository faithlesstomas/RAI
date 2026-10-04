"""Published action contracts remain synchronized with the runtime."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rai.actions.records import ActionProposal
from rai.actions.schemas import action_json_schema
from rai.kernel.records import ProducerIdentity


def test_published_action_schema_matches_runtime():
    assert json.loads(Path('schemas/rai.actions.v1.schema.json').read_text()) == action_json_schema()


def test_proposal_cannot_carry_model_granted_approval():
    values = dict(producer=ProducerIdentity(producer_id='user', kind='human', version='1.0.0'),
                  source_turn_id='turn', task_id='task', capability='application.launch',
                  resource_handle='opaque')
    proposal = ActionProposal(**values)
    assert ActionProposal.model_validate_json(proposal.model_dump_json()) == proposal
    with pytest.raises(ValidationError):
        ActionProposal(**values, approved=True)
