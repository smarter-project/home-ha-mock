# home-ha-mock

This chart deploys home assistant emulator

For more information on smarter go to https://getsmarter.io

## TL;DR

```console
helm install --create-namespace --namespace <namespace to use>  home-ha-mock home-ha-mock
```

# Overview

The ha-mock emulates a home assistant generating events to exercise the demo

# Prerequisites

This chart assumes a full deployment of k3s with traefik, etc.

* k3s 1.25+
* Helm 3.2.0+

# Uninstalling the Chart

```
helm delete home-ha-mock --namespace <namespace to use>
```

# Parameters

## Common parameters

| Name | Description | Value |
| ---- | ----------- | ----- |
