from collections import defaultdict


def domain_expert_features_lists(outcome_name: str, treatment0_name, treatment1_name, cur_treatment) -> list:
    """
    returns specific features to include in model for outcome~treatment combination
    :param outcome_name:
    :param treatment0_name: value of treatment 0 (can be string or number)
    :param treatment1_name: value of treatment 1 (can be string or number)
    :param cur_treatment: arm to return feature list for
    :return: list of domain expert feature names to manually add after feature selection
    """

    domain_expert_features_dict = defaultdict(dict)
    domain_expert_features_dict[("y", 0, 1)][1] = ['W0', 'W1', 'X0']
    domain_expert_features_dict[("y", 0, 1)][0] = ['W1', 'X0']
    try:
        domain_expert_features = domain_expert_features_dict[
            (outcome_name, treatment0_name, treatment1_name)][cur_treatment]
    except KeyError:
        print("no domain expert features for combination {} {} {} - returning default list")
        domain_expert_features = ['W0', 'W1', 'X0']
    return domain_expert_features
